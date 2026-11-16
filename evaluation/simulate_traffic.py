import os
import sys
import json
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.embeddings import EmbeddingModel
from src.cache import SemanticCache
from src.llm import LLMCaller
from src.service import FAQService

def generate_zipf_traffic(num_requests: int = 500, alpha: float = 1.25) -> list:
    """
    Generates a synthetic query stream where repeated questions follow a power-law (Zipf) distribution.
    Uses positive paraphrases from data/eval_dataset_100.json along with occasional hard negatives.
    """
    with open("data/eval_dataset_100.json", "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    # 60 Positives + 40 Negatives
    positives = [e["question"] for e in eval_items if e["faq_id"] != "none"]
    negatives = [e["question"] for e in eval_items if e["faq_id"] == "none"]

    # Top 30 most frequent questions receive Zipf probabilities
    n_freq = len(positives)
    ranks = np.arange(1, n_freq + 1)
    weights = 1.0 / (ranks ** alpha)
    weights /= weights.sum()

    # Generate 500 stream queries: 85% from repeated positive pool, 15% from hard negatives/novel
    np.random.seed(42)
    stream = []
    for _ in range(num_requests):
        if np.random.rand() < 0.85:
            # Pick from positive pool using Zipf distribution
            chosen_q = np.random.choice(positives, p=weights)
        else:
            # Pick from novel/negative pool uniformly
            chosen_q = np.random.choice(negatives)
        stream.append(chosen_q)

    return stream

def run_simulation(stream: list, warm_start: bool, embedder: EmbeddingModel, threshold: float = 0.83) -> dict:
    cache = SemanticCache(dimension=384, read_only=False)
    if warm_start:
        cache.warm_from_faqs("data/faqs_30.json", embedder, current_time=0.0)

    llm = LLMCaller(mode="mock")
    service = FAQService(cache=cache, embedder=embedder, llm_caller=llm, threshold=threshold, use_guard=True)

    latencies = []
    baseline_latencies = []
    cache_hits = 0
    total_tokens_with_cache = 0
    total_tokens_without_cache = 0

    # Advance virtual clock by 5 to 30 seconds per request
    virtual_time = 0.0

    for i, query in enumerate(stream):
        virtual_time += np.random.uniform(5.0, 30.0)
        
        # Service call with virtual clock
        res = service.ask(query, current_time=virtual_time, virtual_clock=True)
        lat = res["latency_ms"]
        latencies.append(lat)

        tokens = res["tokens_used"]
        total_tokens_with_cache += tokens

        if res["hit"]:
            cache_hits += 1

        # Baseline: simulate pure LLM call without cache
        baseline_llm = llm._call_mock_llm(query, virtual_clock=True)
        baseline_lat = baseline_llm["latency_ms"]
        baseline_latencies.append(baseline_lat)
        total_tokens_without_cache += baseline_llm["total_tokens"]

    n = len(stream)
    hit_rate = (cache_hits / n) * 100.0
    llm_calls_saved = cache_hits
    llm_call_reduction_pct = (llm_calls_saved / n) * 100.0
    token_reduction_pct = ((total_tokens_without_cache - total_tokens_with_cache) / total_tokens_without_cache) * 100.0

    return {
        "mode": "Warm Start" if warm_start else "Cold Start",
        "total_requests": n,
        "cache_hits": cache_hits,
        "hit_rate_pct": round(hit_rate, 2),
        "llm_calls_made": n - cache_hits,
        "llm_call_reduction_pct": round(llm_call_reduction_pct, 2),
        "token_reduction_pct": round(token_reduction_pct, 2),
        "latencies_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "avg": round(float(np.mean(latencies)), 2)
        },
        "baseline_latencies_ms": {
            "p50": round(float(np.percentile(baseline_latencies, 50)), 2),
            "p95": round(float(np.percentile(baseline_latencies, 95)), 2),
            "p99": round(float(np.percentile(baseline_latencies, 99)), 2),
            "avg": round(float(np.mean(baseline_latencies)), 2)
        }
    }

def main():
    print("Generating 500 Zipfian requests stream...")
    stream = generate_zipf_traffic(500, alpha=1.25)

    print("Loading embedding model...")
    embedder = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("\nRunning Cold Start Simulation (Empty Cache + Online Fill)...")
    cold_results = run_simulation(stream, warm_start=False, embedder=embedder, threshold=0.83)

    print("\nRunning Warm Start Simulation (Pre-loaded with 30 FAQs)...")
    warm_results = run_simulation(stream, warm_start=True, embedder=embedder, threshold=0.83)

    simulation_summary = {
        "cold_start": cold_results,
        "warm_start": warm_results
    }

    os.makedirs("results", exist_ok=True)
    with open("results/traffic_simulation.json", "w", encoding="utf-8") as f:
        json.dump(simulation_summary, f, indent=2)

    print("\n" + "="*80)
    print(" 500-REQUEST ZIPFIAN TRAFFIC SIMULATION (VIRTUAL CLOCK)")
    print("="*80)
    print(f"{'Metric':<32} | {'Cold Start':<18} | {'Warm Start':<18}")
    print("-"*80)
    print(f"{'Total Requests':<32} | {cold_results['total_requests']:<18} | {warm_results['total_requests']:<18}")
    print(f"{'Cache Hits':<32} | {cold_results['cache_hits']:<18} | {warm_results['cache_hits']:<18}")
    print(f"{'Hit Rate':<32} | {cold_results['hit_rate_pct']}%{'':<13} | {warm_results['hit_rate_pct']}%{'':<13}")
    print(f"{'LLM Calls Reduced':<32} | {cold_results['llm_call_reduction_pct']}%{'':<13} | {warm_results['llm_call_reduction_pct']}%{'':<13}")
    print(f"{'Tokens Reduced':<32} | {cold_results['token_reduction_pct']}%{'':<13} | {warm_results['token_reduction_pct']}%{'':<13}")
    print("-"*80)
    print(f"{'Baseline Latency p50 / p95':<32} | {cold_results['baseline_latencies_ms']['p50']}ms / {cold_results['baseline_latencies_ms']['p95']}ms{'':<5} | {warm_results['baseline_latencies_ms']['p50']}ms / {warm_results['baseline_latencies_ms']['p95']}ms")
    print(f"{'Cached Latency p50 / p95':<32} | {cold_results['latencies_ms']['p50']}ms / {cold_results['latencies_ms']['p95']}ms{'':<5} | {warm_results['latencies_ms']['p50']}ms / {warm_results['latencies_ms']['p95']}ms")
    print(f"{'Average Latency (Before -> After)':<32} | {cold_results['baseline_latencies_ms']['avg']}ms -> {cold_results['latencies_ms']['avg']}ms | {warm_results['baseline_latencies_ms']['avg']}ms -> {warm_results['latencies_ms']['avg']}ms")
    print("="*80)
    print("Simulation completed in < 2 seconds using virtual clock.")

if __name__ == "__main__":
    main()
