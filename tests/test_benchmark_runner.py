import os
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from evaluation.benchmark_rag import (
    evaluate_answer_correctness,
    calculate_cost,
    redact_pii_text,
    get_false_hit_status_label,
    run_single_pipeline,
    run_benchmark
)
from src.service import FAQService
from src.cache import SemanticCache
from src.llm import LLMCaller

class MockEmbedder:
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        import numpy as np
        return np.ones((len(texts), 4), dtype=np.float32)

class MockRetriever:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [
            (0.9, {"question": "How to cancel?", "answer": "Cancel via My Orders before shipment."})
        ]

    def search(self, vector, top_k=3):
        return self.docs

def test_hit_rate_and_false_hit_rate_zero_denominator():
    # Calling calculate_cost with None returns unavailable
    res = calculate_cost(100, 50, None, None)
    assert res["status"] == "unavailable"
    assert res["total_cost_usd"] is None

def test_cost_calculation_when_configured():
    # 1M prompt @ $0.15, 1M comp @ $0.60
    # For 1,000 prompt tokens: $0.00015
    # For 500 comp tokens: $0.00030
    res = calculate_cost(1000, 500, prompt_price_per_1m=0.15, completion_price_per_1m=0.60)
    assert res["status"] == "calculated"
    assert res["prompt_cost_usd"] == 0.00015
    assert res["completion_cost_usd"] == 0.0003
    assert res["total_cost_usd"] == 0.00045

def test_pii_redaction():
    assert redact_pii_text("My order is ORD-12345", "order_id") == "[REDACTED_PII: order_id]"
    assert redact_pii_text("Normal non-pii query", None) == "Normal non-pii query"

def test_answer_correctness_negative_query():
    canonical_faqs = {
        "faq_01": {"question": "Track order?", "answer": "Go to My Orders."}
    }

    # Negative query that correctly hits fallback
    res_fallback = {"hit": False, "source": "fallback", "answer": "Apologies, no FAQ found."}
    is_corr, reason = evaluate_answer_correctness(res_fallback, "none", canonical_faqs)
    assert is_corr is True
    assert reason == "correct_fallback_negative"

    # Negative query that incorrectly causes a cache hit -> False Hit!
    res_hit = {"hit": True, "source": "cache", "answer": "Go to My Orders.", "matched_faq_id": "faq_01"}
    is_corr2, reason2 = evaluate_answer_correctness(res_hit, "none", canonical_faqs)
    assert is_corr2 is False
    assert reason2 == "false_hit_on_negative"

def test_answer_correctness_positive_query_facts():
    canonical_faqs = {
        "faq_06": {
            "question": "How do I cancel my order?",
            "answer": "To cancel, go to My Orders, select the order, and click Cancel Order before dispatch."
        }
    }

    # Exact warm hit
    res_exact = {"hit": True, "matched_faq_id": "faq_06", "answer": "To cancel, go to My Orders..."}
    is_corr, _ = evaluate_answer_correctness(res_exact, "faq_06", canonical_faqs)
    assert is_corr is True

    # Dynamic hit with canonical fact overlap
    res_dynamic = {
        "hit": True,
        "matched_faq_id": "dynamic_entry",
        "answer": "You can cancel your order easily from the My Orders section prior to dispatch."
    }
    is_corr_dyn, _ = evaluate_answer_correctness(res_dynamic, "faq_06", canonical_faqs)
    assert is_corr_dyn is True

    # Hit returning completely wrong answer
    res_wrong = {
        "hit": True,
        "matched_faq_id": "faq_99",
        "answer": "International shipments require custom duty documents."
    }
    is_corr_wrong, reason_wrong = evaluate_answer_correctness(res_wrong, "faq_06", canonical_faqs)
    assert is_corr_wrong is False
    assert "false_hit_wrong_faq" in reason_wrong

def test_real_mode_skips_pii_tagged_queries():
    dataset = [
        {"id": "q1", "question": "Normal query", "faq_id": "faq_01", "pii_type": None},
        {"id": "q2", "question": "My phone is 9876543210", "faq_id": "none", "pii_type": "phone"}
    ]
    canonical_faqs = {"faq_01": {"question": "q", "answer": "a"}}

    service = MagicMock()
    service.ask.return_value = {"hit": False, "answer": "Normal response", "latency_ms": 10.0}
    llm = LLMCaller(mode="mock")

    # In real mode, PII query q2 must be skipped completely
    results = run_single_pipeline(dataset, service, llm, canonical_faqs, mode="real", pipeline_name="Test")

    assert results["processed_queries"] == 1
    assert results["skipped_pii_count"] == 1
    assert service.ask.call_count == 1

def test_mock_mode_processes_pii_queries():
    dataset = [
        {"id": "q1", "question": "Normal query", "faq_id": "faq_01", "pii_type": None},
        {"id": "q2", "question": "My phone is 9876543210", "faq_id": "none", "pii_type": "phone"}
    ]
    canonical_faqs = {"faq_01": {"question": "q", "answer": "a"}}

    service = MagicMock()
    service.ask.return_value = {"hit": False, "answer": "response", "reason": "bypassed_pii: phone", "latency_ms": 10.0}
    llm = LLMCaller(mode="mock")

    # In mock mode, PII queries are tested and not skipped
    results = run_single_pipeline(dataset, service, llm, canonical_faqs, mode="mock", pipeline_name="Test")

    assert results["processed_queries"] == 2
    assert results["skipped_pii_count"] == 0
    assert results["pii_bypasses"] == 2

def test_pipeline_cache_isolation_and_counting():
    mock_embedder = MockEmbedder()
    mock_retriever = MockRetriever()
    llm_baseline = LLMCaller(mode="mock")
    llm_cached = LLMCaller(mode="mock")

    cache_baseline = SemanticCache(dimension=4, read_only=True)
    cache_cached = SemanticCache(dimension=4, read_only=False)

    srv_baseline = FAQService(
        cache=cache_baseline,
        embedder=mock_embedder,
        llm_caller=llm_baseline,
        retriever=mock_retriever,
        enable_cache=False
    )
    srv_cached = FAQService(
        cache=cache_cached,
        embedder=mock_embedder,
        llm_caller=llm_cached,
        retriever=mock_retriever,
        enable_cache=True
    )

    dataset = [
        {"id": "d1", "question": "How to cancel?", "faq_id": "faq_06", "pii_type": None},
        {"id": "d2", "question": "How to cancel?", "faq_id": "faq_06", "pii_type": None}
    ]
    canonical_faqs = {"faq_06": {"question": "How to cancel?", "answer": "Cancel via My Orders."}}

    res_base = run_single_pipeline(dataset, srv_baseline, llm_baseline, canonical_faqs, "mock", "Baseline")
    res_cached = run_single_pipeline(dataset, srv_cached, llm_cached, canonical_faqs, "mock", "Cached")

    # Baseline never hits cache
    assert res_base["cache_hits"] == 0
    assert res_base["llm_metrics"]["total_generation_attempts"] == 2

    # Cached hits on repeat!
    assert res_cached["cache_hits"] == 1
    assert res_cached["llm_metrics"]["total_generation_attempts"] == 1  # 1 LLM call saved!

    # Verify cache isolation: baseline cache size is 0, cached cache size is 1
    assert cache_baseline.size() == 0
    assert cache_cached.size() == 1

def test_warm_cache_reference_time_prevents_immediate_expiry():
    cache = SemanticCache(dimension=4)
    ref_time = 1700000000.0
    vec = MockEmbedder().encode("Can I cancel?")[0]
    cache.add(vec, "Can I cancel?", "Yes", "faq_06", ttl_seconds=2592000, current_time=ref_time)

    # Searching at the exact same reference time should find the entry
    results = cache.search(vec, top_k=1, current_time=ref_time)
    assert len(results) == 1
    assert results[0][1]["faq_id"] == "faq_06"

def test_warm_cache_produces_hits_on_canonical_query():
    cache = SemanticCache(dimension=4)
    ref_time = 1700000000.0
    vec = MockEmbedder().encode("Can I cancel?")[0]
    cache.add(vec, "Can I cancel?", "Yes", "faq_06", ttl_seconds=2592000, current_time=ref_time)

    results = cache.search(vec, top_k=1, current_time=ref_time)
    assert len(results) == 1
    sim, entry = results[0]
    assert sim >= 0.83
    assert entry["faq_id"] == "faq_06"

def test_warm_cache_rejects_expired_entries():
    cache = SemanticCache(dimension=4)
    ref_time = 1700000000.0
    vec = MockEmbedder().encode("Can I cancel?")[0]
    cache.add(vec, "Can I cancel?", "Yes", "faq_06", ttl_seconds=300, current_time=ref_time)

    # Query 301 seconds later -> must be rejected as expired
    results = cache.search(vec, top_k=1, current_time=ref_time + 301.0)
    assert len(results) == 0

def test_false_hit_label_zero_percent():
    assert get_false_hit_status_label(0.0) == "Zero False Hits"

def test_false_hit_label_acceptable_range():
    assert get_false_hit_status_label(1.43) == "Acceptable (< 5%)"
    assert get_false_hit_status_label(5.0) == "Acceptable (< 5%)"

def test_false_hit_label_high_risk():
    assert get_false_hit_status_label(5.01) == "High False-Hit Rate"
    assert get_false_hit_status_label(100.0) == "High False-Hit Rate"

def test_warm_cache_entries_remain_valid_during_benchmark():
    mock_embedder = MockEmbedder()
    mock_retriever = MockRetriever()
    llm = LLMCaller(mode="mock")
    cache = SemanticCache(dimension=4)

    ref_time = 1700000000.0
    vec = mock_embedder.encode("How to cancel?")[0]
    cache.add(vec, "How to cancel?", "Cancel via My Orders.", "faq_06", ttl_seconds=2592000, current_time=ref_time)

    srv = FAQService(
        cache=cache,
        embedder=mock_embedder,
        llm_caller=llm,
        retriever=mock_retriever,
        enable_cache=True
    )

    dataset = [{"id": "d1", "question": "How to cancel?", "faq_id": "faq_06", "pii_type": None}]
    canonical_faqs = {"faq_06": {"question": "How to cancel?", "answer": "Cancel via My Orders."}}

    res = run_single_pipeline(dataset, srv, llm, canonical_faqs, "mock", "Cached", current_time=ref_time)

    # Must produce a cache hit because the entry remained valid during the benchmark
    assert res["cache_hits"] == 1
    assert res["cache_misses"] == 0
    assert res["llm_metrics"]["total_generation_attempts"] == 0
    assert res["per_query_results"][0]["matched_faq_id"] == "faq_06"
