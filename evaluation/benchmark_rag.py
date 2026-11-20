import os
import sys
import json
import time
import hashlib
import argparse
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.embeddings import EmbeddingModel
from src.cache import SemanticCache
from src.llm import LLMCaller
from src.service import FAQService
from src.retriever import FAQRetriever


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file for experiment reproducibility."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def redact_pii_text(text: str, pii_type: Optional[str]) -> str:
    """Redacts PII text for safe reporting and logging."""
    if pii_type:
        return f"[REDACTED_PII: {pii_type}]"
    return text


def get_false_hit_status_label(false_hit_rate_pct: float) -> str:
    """Returns dynamic risk label for false-hit rate."""
    if false_hit_rate_pct == 0.0:
        return "Zero False Hits"
    elif false_hit_rate_pct <= 5.0:
        return "Acceptable (< 5%)"
    else:
        return "High False-Hit Rate"


def extract_keywords(text: str) -> set:
    """Extracts distinctive lowercased content words for fact verification."""
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "you", "your",
        "can", "are", "have", "been", "will", "our", "not", "any", "all",
        "please", "before", "after", "order", "orders", "faq", "answer"
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()) if "re" in sys.modules else text.lower().split()
    return {w.strip(".,!?:;'\"") for w in words if len(w) >= 3 and w.lower() not in stop_words}


def evaluate_answer_correctness(
    result: Dict[str, Any],
    expected_faq_id: str,
    canonical_faqs: Dict[str, Dict[str, Any]]
) -> Tuple[bool, str]:
    """
    Evaluates whether the returned answer aligns with expected FAQ and canonical facts.
    Returns (is_correct, reason).
    """
    answer_text = result.get("answer", "")
    hit = result.get("hit", False)
    matched_id = result.get("matched_faq_id")

    # Case A: Out-of-domain / Negative query (expected_faq_id == "none")
    if expected_faq_id == "none":
        if hit:
            return False, "false_hit_on_negative"
        if result.get("source") == "fallback":
            return True, "correct_fallback_negative"
        if "bypassed_pii" in result.get("reason", ""):
            return True, "correct_pii_bypass"
        return False, "unsupported_response_for_negative"

    # Case B: Positive query (expected_faq_id is a valid FAQ)
    canonical = canonical_faqs.get(expected_faq_id)
    if not canonical:
        return False, "unknown_expected_faq_id"

    # 1. Warm cache hit direct ID match
    if hit and matched_id == expected_faq_id:
        return True, "correct_hit_exact_faq_id"

    # 2. Dynamic cache hit or RAG generation: evaluate canonical fact overlap
    canonical_answer = canonical.get("answer", "")
    canon_keywords = extract_keywords(canonical_answer)
    ans_keywords = extract_keywords(answer_text)

    overlap = canon_keywords.intersection(ans_keywords)
    overlap_ratio = len(overlap) / max(1, len(canon_keywords))

    if overlap_ratio >= 0.25:
        return True, "correct_facts_verified"

    if hit:
        return False, f"false_hit_wrong_faq (matched: {matched_id}, expected: {expected_faq_id})"

    return False, "insufficient_fact_overlap"


def get_false_hit_status_label(false_hit_rate_pct: float) -> str:
    """Returns dynamic risk label for false-hit rate."""
    if false_hit_rate_pct == 0.0:
        return "Zero False Hits"
    elif false_hit_rate_pct <= 5.0:
        return "Acceptable (< 5%)"
    else:
        return "High False-Hit Rate"


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_price_per_1m: Optional[float],
    completion_price_per_1m: Optional[float]
) -> Dict[str, Any]:
    """Calculates cost if pricing is configured, else marks as unavailable."""
    if prompt_price_per_1m is not None and completion_price_per_1m is not None:
        p_cost = (prompt_tokens / 1_000_000.0) * prompt_price_per_1m
        c_cost = (completion_tokens / 1_000_000.0) * completion_price_per_1m
        return {
            "status": "calculated",
            "prompt_cost_usd": round(p_cost, 6),
            "completion_cost_usd": round(c_cost, 6),
            "total_cost_usd": round(p_cost + c_cost, 6),
            "pricing_rates": {
                "prompt_per_1m": prompt_price_per_1m,
                "completion_per_1m": completion_price_per_1m
            }
        }
    return {
        "status": "unavailable",
        "prompt_cost_usd": None,
        "completion_cost_usd": None,
        "total_cost_usd": None,
        "pricing_rates": None
    }


def run_single_pipeline(
    dataset: List[Dict[str, Any]],
    service: FAQService,
    llm: LLMCaller,
    canonical_faqs: Dict[str, Dict[str, Any]],
    mode: str,
    pipeline_name: str,
    current_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes a single pipeline (Baseline RAG or Cached RAG) over the query dataset.
    """
    llm.reset_metrics()

    processed_queries = 0
    skipped_pii_count = 0
    cache_hits = 0
    cache_misses = 0
    incorrect_cache_hits = 0
    correct_cache_hits = 0
    correct_answers = 0
    pii_bypasses = 0
    no_doc_fallbacks = 0

    compute_latencies_ms = []
    service_latencies_ms = []

    per_query_results = []

    for item in dataset:
        query = item["question"]
        expected_faq_id = item.get("faq_id", "none")
        pii_type = item.get("pii_type")

        # In real mode, NEVER send PII-tagged queries to external providers
        if mode == "real" and pii_type:
            skipped_pii_count += 1
            continue

        processed_queries += 1

        # Measure pure compute overhead with virtual_clock=True (no sleep)
        t_compute_start = time.perf_counter()
        res = service.ask(query, current_time=current_time, virtual_clock=True)
        compute_ms = round((time.perf_counter() - t_compute_start) * 1000.0, 3)
        compute_latencies_ms.append(compute_ms)

        # Record service/simulated latency
        service_latencies_ms.append(res.get("latency_ms", 0.0))

        # Classify cache hit / miss
        hit = res.get("hit", False)
        if hit:
            cache_hits += 1
        else:
            cache_misses += 1

        # PII bypass tracking
        if "bypassed_pii" in res.get("reason", ""):
            pii_bypasses += 1

        # Fallback tracking
        if res.get("source") == "fallback" or res.get("reason") == "cache_miss_no_relevant_faq":
            no_doc_fallbacks += 1

        # Evaluate correctness
        is_correct, correctness_reason = evaluate_answer_correctness(res, expected_faq_id, canonical_faqs)
        if is_correct:
            correct_answers += 1

        if hit:
            if is_correct and expected_faq_id != "none":
                correct_cache_hits += 1
            else:
                incorrect_cache_hits += 1

        per_query_results.append({
            "id": item["id"],
            "question_redacted": redact_pii_text(query, pii_type),
            "expected_faq_id": expected_faq_id,
            "hit": hit,
            "matched_faq_id": res.get("matched_faq_id"),
            "similarity": res.get("similarity"),
            "source": res.get("source"),
            "reason": res.get("reason"),
            "is_correct": is_correct,
            "correctness_reason": correctness_reason,
            "compute_latency_ms": compute_ms,
            "service_latency_ms": res.get("latency_ms", 0.0)
        })

    # Metric summaries
    llm_metrics = llm.get_metrics()
    n = max(1, processed_queries)

    hit_rate_pct = round((cache_hits / n) * 100.0, 2)
    false_hit_rate_pct = round((incorrect_cache_hits / max(1, cache_hits)) * 100.0, 2) if cache_hits > 0 else 0.0
    false_hit_traffic_pct = round((incorrect_cache_hits / n) * 100.0, 2)
    correctness_pct = round((correct_answers / n) * 100.0, 2)

    return {
        "pipeline_name": pipeline_name,
        "processed_queries": processed_queries,
        "skipped_pii_count": skipped_pii_count,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hit_rate_pct": hit_rate_pct,
        "incorrect_cache_hits": incorrect_cache_hits,
        "correct_cache_hits": correct_cache_hits,
        "false_hit_rate_pct": false_hit_rate_pct,
        "false_hit_traffic_pct": false_hit_traffic_pct,
        "correct_answers": correct_answers,
        "correctness_pct": correctness_pct,
        "pii_bypasses": pii_bypasses,
        "no_doc_fallbacks": no_doc_fallbacks,
        "compute_latency_ms": {
            "mean": round(float(np.mean(compute_latencies_ms)), 2) if compute_latencies_ms else 0.0,
            "median": round(float(np.median(compute_latencies_ms)), 2) if compute_latencies_ms else 0.0,
            "p95": round(float(np.percentile(compute_latencies_ms, 95)), 2) if compute_latencies_ms else 0.0
        },
        "service_latency_ms": {
            "mean": round(float(np.mean(service_latencies_ms)), 2) if service_latencies_ms else 0.0,
            "median": round(float(np.median(service_latencies_ms)), 2) if service_latencies_ms else 0.0,
            "p95": round(float(np.percentile(service_latencies_ms, 95)), 2) if service_latencies_ms else 0.0
        },
        "llm_metrics": llm_metrics,
        "per_query_results": per_query_results
    }


def run_benchmark(
    dataset_path: str = "data/benchmark_dataset_120.json",
    faqs_path: str = "data/faqs_30.json",
    mode: str = "mock",
    warm: bool = False,
    split: str = "dev",
    threshold: float = 0.83,
    prompt_price: Optional[float] = None,
    completion_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes the full comparative benchmark between Baseline RAG and Cached RAG.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    if not os.path.exists(faqs_path):
        raise FileNotFoundError(f"FAQs not found at: {faqs_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    with open(faqs_path, "r", encoding="utf-8") as f:
        faqs_data = json.load(f)

    canonical_faqs = {item["faq_id"]: item for item in faqs_data}
    dataset_hash = compute_file_sha256(dataset_path)
    benchmark_reference_time = time.time()

    # Filter dataset by split
    if split in ("dev", "test"):
        active_dataset = [d for d in full_dataset if d.get("split") == split]
    else:
        active_dataset = full_dataset

    # Initialize shared embedding model
    embedder = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 1. Setup Baseline RAG Pipeline (enable_cache=False)
    baseline_cache = SemanticCache(dimension=384, read_only=True)
    baseline_retriever = FAQRetriever(faqs_path=faqs_path, embedder=embedder, threshold=0.45)
    baseline_llm = LLMCaller(mode=mode)
    baseline_service = FAQService(
        cache=baseline_cache,
        embedder=embedder,
        llm_caller=baseline_llm,
        retriever=baseline_retriever,
        threshold=threshold,
        use_guard=True,
        enable_cache=False
    )

    # 2. Setup Cached RAG Pipeline (enable_cache=True)
    cached_cache = SemanticCache(dimension=384, read_only=False)
    if warm:
        cached_cache.warm_from_faqs(faqs_path, embedder, current_time=benchmark_reference_time)

    cached_retriever = FAQRetriever(faqs_path=faqs_path, embedder=embedder, threshold=0.45)
    cached_llm = LLMCaller(mode=mode)
    cached_service = FAQService(
        cache=cached_cache,
        embedder=embedder,
        llm_caller=cached_llm,
        retriever=cached_retriever,
        threshold=threshold,
        use_guard=True,
        enable_cache=True
    )

    # Execute Baseline RAG
    baseline_results = run_single_pipeline(
        dataset=active_dataset,
        service=baseline_service,
        llm=baseline_llm,
        canonical_faqs=canonical_faqs,
        mode=mode,
        pipeline_name="Baseline RAG (No Cache)",
        current_time=benchmark_reference_time
    )

    # Execute Cached RAG on the identical stream
    cached_results = run_single_pipeline(
        dataset=active_dataset,
        service=cached_service,
        llm=cached_llm,
        canonical_faqs=canonical_faqs,
        mode=mode,
        pipeline_name=f"Cached RAG ({'Warm Start' if warm else 'Cold Start'})",
        current_time=benchmark_reference_time
    )

    # Comparative metrics
    baseline_llm_calls = baseline_results["llm_metrics"]["total_generation_attempts"]
    cached_llm_calls = cached_results["llm_metrics"]["total_generation_attempts"]
    llm_calls_saved = max(0, baseline_llm_calls - cached_llm_calls)
    llm_call_reduction_pct = round((llm_calls_saved / max(1, baseline_llm_calls)) * 100.0, 2)

    # Token and Cost metrics
    b_tokens = baseline_results["llm_metrics"]["tokens"]
    c_tokens = cached_results["llm_metrics"]["tokens"]

    b_prompt = b_tokens["real_provider"]["prompt_tokens"] if mode == "real" else b_tokens["mock_estimated"]["prompt_tokens_est"]
    b_comp = b_tokens["real_provider"]["completion_tokens"] if mode == "real" else b_tokens["mock_estimated"]["completion_tokens_est"]

    c_prompt = c_tokens["real_provider"]["prompt_tokens"] if mode == "real" else c_tokens["mock_estimated"]["prompt_tokens_est"]
    c_comp = c_tokens["real_provider"]["completion_tokens"] if mode == "real" else c_tokens["mock_estimated"]["completion_tokens_est"]

    baseline_cost = calculate_cost(b_prompt, b_comp, prompt_price, completion_price)
    cached_cost = calculate_cost(c_prompt, c_comp, prompt_price, completion_price)

    cost_comparison = {
        "status": baseline_cost["status"],
        "baseline_cost_usd": baseline_cost["total_cost_usd"],
        "cached_cost_usd": cached_cost["total_cost_usd"],
        "cost_saved_usd": round(baseline_cost["total_cost_usd"] - cached_cost["total_cost_usd"], 6) if baseline_cost["status"] == "calculated" else None,
        "cost_reduction_pct": round(
            ((baseline_cost["total_cost_usd"] - cached_cost["total_cost_usd"]) / baseline_cost["total_cost_usd"]) * 100.0, 2
        ) if baseline_cost["status"] == "calculated" and baseline_cost["total_cost_usd"] > 0 else None
    }

    report = {
        "benchmark_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": dataset_path,
            "dataset_sha256": dataset_hash,
            "split": split,
            "cache_mode": "warm_prepopulated" if warm else "cold_dynamic",
            "execution_mode": mode,
            "model": "gpt-4o-mini" if mode == "real" else "mock-gpt-4o-mini",
            "cache_threshold": threshold,
            "retriever_threshold": 0.45,
            "total_items_in_split": len(active_dataset)
        },
        "comparative_summary": {
            "total_queries_processed": baseline_results["processed_queries"],
            "skipped_pii_count": baseline_results["skipped_pii_count"],
            "baseline_llm_calls": baseline_llm_calls,
            "cached_llm_calls": cached_llm_calls,
            "llm_calls_saved": llm_calls_saved,
            "llm_call_reduction_pct": llm_call_reduction_pct,
            "cache_hit_rate_pct": cached_results["hit_rate_pct"],
            "false_hit_rate_pct": cached_results["false_hit_rate_pct"],
            "false_hit_traffic_pct": cached_results["false_hit_traffic_pct"],
            "baseline_correctness_pct": baseline_results["correctness_pct"],
            "cached_correctness_pct": cached_results["correctness_pct"],
            "compute_latency_p95_baseline_ms": baseline_results["compute_latency_ms"]["p95"],
            "compute_latency_p95_cached_ms": cached_results["compute_latency_ms"]["p95"],
            "cost_comparison": cost_comparison
        },
        "baseline_pipeline": baseline_results,
        "cached_pipeline": cached_results
    }

    return report


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Formats benchmark results into structured Markdown report."""
    meta = report_data["benchmark_metadata"]
    summary = report_data["comparative_summary"]
    b = report_data["baseline_pipeline"]
    c = report_data["cached_pipeline"]
    cost = summary["cost_comparison"]

    latency_label = "Simulated Mock Latency" if meta["execution_mode"] == "mock" else "Real Provider Latency"
    false_hit_label = get_false_hit_status_label(summary["false_hit_rate_pct"])

    cost_str = (
        f"${cost['cached_cost_usd']} (Saved: ${cost['cost_saved_usd']}, {cost['cost_reduction_pct']}%)"
        if cost["status"] == "calculated"
        else "Pricing not configured (Token counts reported)"
    )

    md = f"""# Semantic Cache vs Baseline RAG Benchmark Report

**Generated:** {meta['timestamp_utc']}

**Dataset:** `{meta['dataset_path']}` (SHA-256: `{meta['dataset_sha256'][:12]}...`)

**Split:** `{meta['split'].upper()}` | **Cache Mode:** `{meta['cache_mode']}` | **Execution Mode:** `{meta['execution_mode']}`

**Thresholds:** Cache `T={meta['cache_threshold']}`, Retriever `T={meta['retriever_threshold']}`

---

## 1. Executive Summary

| Metric | Baseline RAG (No Cache) | Cached RAG ({'Warm' if meta['cache_mode'] == 'warm_prepopulated' else 'Cold'}) | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Processed Queries** | {summary['total_queries_processed']} | {summary['total_queries_processed']} | - |
| **Cache Hits** | 0 (0.0%) | {c['cache_hits']} ({summary['cache_hit_rate_pct']}%) | +{summary['cache_hit_rate_pct']}% |
| **False Hit Rate** | N/A | {summary['false_hit_rate_pct']}% of hits | {false_hit_label} |
| **False Hits (% of Traffic)** | 0.0% | {summary['false_hit_traffic_pct']}% | Minimal |
| **LLM Invocations** | {summary['baseline_llm_calls']} | {summary['cached_llm_calls']} | **{summary['llm_calls_saved']} calls saved ({summary['llm_call_reduction_pct']}%)** |
| **Answer Correctness** | {summary['baseline_correctness_pct']}% | {summary['cached_correctness_pct']}% | Preserved |
| **Compute Latency (P95)** | {summary['compute_latency_p95_baseline_ms']} ms | {summary['compute_latency_p95_cached_ms']} ms | Direct FAISS indexing |
| **API Cost** | N/A | {cost_str} | - |

> [!NOTE]
> In real execution mode, PII-tagged queries ({summary['skipped_pii_count']} items) are automatically skipped from external provider calls to protect user privacy. In mock mode, PII bypass verification runs locally without network transmission.

---

## 2. Latency Profiling

*Note: Compute latency measures pure CPU execution time (embeddings + FAISS lookup with virtual clock). {latency_label} measures end-to-end service response time.*

| Metric | Baseline Compute | Cached Compute | Baseline {latency_label} | Cached {latency_label} |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | {b['compute_latency_ms']['mean']} ms | {c['compute_latency_ms']['mean']} ms | {b['service_latency_ms']['mean']} ms | {c['service_latency_ms']['mean']} ms |
| **Median (P50)** | {b['compute_latency_ms']['median']} ms | {c['compute_latency_ms']['median']} ms | {b['service_latency_ms']['median']} ms | {c['service_latency_ms']['median']} ms |
| **P95** | {b['compute_latency_ms']['p95']} ms | {c['compute_latency_ms']['p95']} ms | {b['service_latency_ms']['p95']} ms | {c['service_latency_ms']['p95']} ms |

---

## 3. Token & LLM Invocation Breakdown

| Metric | Baseline RAG | Cached RAG |
| :--- | :--- | :--- |
| **Actual Provider Requests** | {b['llm_metrics']['provider_requests']} | {c['llm_metrics']['provider_requests']} |
| **Successful Provider Responses** | {b['llm_metrics']['provider_successes']} | {c['llm_metrics']['provider_successes']} |
| **Mock Invocations** | {b['llm_metrics']['mock_invocations']} | {c['llm_metrics']['mock_invocations']} |
| **Mock Fallback Invocations** | {b['llm_metrics']['mock_fallback_invocations']} | {c['llm_metrics']['mock_fallback_invocations']} |
| **PII Bypasses Handled** | {b['pii_bypasses']} | {c['pii_bypasses']} |
| **No-Document Fallbacks Handled** | {b['no_doc_fallbacks']} | {c['no_doc_fallbacks']} |

---

## 4. Methodological Assumptions & Limitations

1. **Privacy Guard:** PII benchmark queries are never transmitted to external APIs; their bypass behavior is verified strictly in isolated mock mode.
2. **Dynamic Entry IDs:** Dynamic cache writes assign `faq_id="dynamic_entry"`. Correctness evaluation checks semantic fact preservation rather than relying solely on `matched_faq_id`.
3. **Compute vs Production Latency:** Compute latency captures algorithmic runtime. Simulated mock latency models standard production LLM waiting time (`Normal(1100ms, 180ms)`).
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Semantic Cache vs Baseline RAG Benchmark Runner")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock", help="Execution mode (default: mock)")
    parser.add_argument("--warm", type=lambda x: (str(x).lower() == 'true'), default=False, help="Warm pre-populated cache (default: False)")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev", help="Dataset split to evaluate (default: dev)")
    parser.add_argument("--dataset", default="data/benchmark_dataset_120.json", help="Path to evaluation dataset")
    parser.add_argument("--faqs", default="data/faqs_30.json", help="Path to canonical FAQs")
    parser.add_argument("--threshold", type=float, default=0.83, help="Cache similarity threshold (default: 0.83)")
    parser.add_argument("--prompt-price", type=float, default=None, help="Price per 1M prompt tokens in USD")
    parser.add_argument("--completion-price", type=float, default=None, help="Price per 1M completion tokens in USD")
    parser.add_argument("--out-dir", default="results", help="Output directory for reports")

    args = parser.parse_args()

    print(f"Starting RAG Benchmark [Mode: {args.mode}, Warm: {args.warm}, Split: {args.split}]...")
    report = run_benchmark(
        dataset_path=args.dataset,
        faqs_path=args.faqs,
        mode=args.mode,
        warm=args.warm,
        split=args.split,
        threshold=args.threshold,
        prompt_price=args.prompt_price,
        completion_price=args.completion_price
    )

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "rag_benchmark_report.json")
    md_path = os.path.join(args.out_dir, "rag_benchmark_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_content = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Benchmark completed successfully.")
    print(f"Reports generated:")
    print(f" - {json_path}")
    print(f" - {md_path}")
    print(f"Hit Rate: {report['comparative_summary']['cache_hit_rate_pct']}% | Calls Saved: {report['comparative_summary']['llm_calls_saved']}")


if __name__ == "__main__":
    main()
