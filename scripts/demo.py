import os
import sys
import time

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.embeddings import EmbeddingModel
from src.cache import SemanticCache
from src.llm import LLMCaller
from src.service import FAQService

def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(f" {title.upper()}")
        print("=" * 80)

def main():
    print_separator("Semantic Cache for FAQ Chatbot - Live Demo")
    print("Initializing embedding model, FAISS index, and mock/real LLM caller...")
    
    embedder = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    cache = SemanticCache(dimension=384, read_only=False)
    llm = LLMCaller()
    service = FAQService(cache=cache, embedder=embedder, llm_caller=llm, threshold=0.83, use_guard=True)
    
    print("Ready! Starting with a clean, empty cache (Cold Start).\n")

    scenarios = [
        {
            "name": "Scenario 1: Cold Query (Cache MISS -> LLM Fallback -> Auto Fill)",
            "query": "How do I cancel my order before it ships?",
            "expectation": "Cache MISS -> Round-trip to LLM (~1000ms) -> Saved to FAISS"
        },
        {
            "name": "Scenario 2: Exact Repeat Question (Instant Cache HIT)",
            "query": "How do I cancel my order before it ships?",
            "expectation": "Cache HIT -> Served in milliseconds (<60ms)"
        },
        {
            "name": "Scenario 3: Semantic Paraphrase (Semantic Cache HIT)",
            "query": "I want to cancel my pending order before it gets shipped",
            "expectation": "Cache HIT -> High cosine similarity (>0.83) -> Served instantly (<60ms)"
        },
        {
            "name": "Scenario 4: Negation Flip / Opposite Polarity (Guarded against False Hit)",
            "query": "order cancel mat karna galti se button press ho gaya tha please continue shipment",
            "expectation": "Negation Guard BLOCKS hit -> Routed safely to LLM (0 False Hits)"
        },
        {
            "name": "Scenario 5: Two-Sided PII Safety Check",
            "query": "Where is my delayed delivery for order ORD-987654321?",
            "expectation": "PII Detected -> Cache search & write BYPASSED (No private data cached)"
        }
    ]

    for sc in scenarios:
        print_separator(sc["name"])
        print(f"User Query: \"{sc['query']}\"")
        print(f"Expected:   {sc['expectation']}")
        print("\nProcessing request...")
        
        start = time.time()
        res = service.ask(sc["query"])
        elapsed = round((time.time() - start) * 1000.0, 2)
        
        print("\n--- Response Received ---")
        print(f"Answer:     {res['answer']}")
        print(f"Status:     {'CACHE HIT' if res['hit'] else 'CACHE MISS'}")
        print(f"Source:     {res['source'].upper()}")
        print(f"Similarity: {res['similarity']}")
        print(f"Tokens:     {res['tokens_used']}")
        print(f"Latency:    {res['latency_ms']} ms (Measured wall-clock: {elapsed} ms)")
        print(f"Reason:     {res['reason']}")
        time.sleep(0.5)

    print_separator("Final Cache Statistics")
    total_entries = cache.size()
    total_hits = sum(e.get("hit_count", 0) for e in cache.entries)
    print(f"Total Entries in Cache: {total_entries}")
    print(f"Total Cache Hits:       {total_hits}")
    print(f"Threshold:              {service.threshold}")
    print(f"Negation Guard:         Enabled")
    print("=" * 80)
    print("Demo completed successfully.\n")

if __name__ == "__main__":
    main()
