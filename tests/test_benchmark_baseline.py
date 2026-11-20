import os
import json
import pytest
from unittest.mock import MagicMock

from src.service import FAQService
from src.guards import contains_pii

class MockEmbedder:
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Return dummy 2D vectors
        import numpy as np
        return np.ones((len(texts), 4), dtype=np.float32)

class MockLLM:
    def __init__(self):
        self.calls = []

    def generate_answer(self, query, virtual_clock=False, context=None):
        self.calls.append({"query": query, "context": context})
        return {
            "answer": "Mock generated answer for testing.",
            "latency_ms": 10.0,
            "total_tokens": 15
        }

class MockRetriever:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [
            (0.85, {"question": "Can I cancel?", "answer": "Yes, before dispatch."})
        ]

    def search(self, vector, top_k=3):
        return self.docs

def test_benchmark_dataset_120_schema_and_pii():
    path = "data/benchmark_dataset_120.json"
    assert os.path.exists(path), "benchmark_dataset_120.json must exist"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 120, "Dataset must have exactly 120 items"

    # Verify first 100 preserve eval_dataset_100.json
    with open("data/eval_dataset_100.json", "r", encoding="utf-8") as f:
        orig = json.load(f)

    for i in range(100):
        assert data[i]["id"] == orig[i]["id"]
        assert data[i]["question"] == orig[i]["question"]
        assert data[i]["split"] == orig[i]["split"]
        assert data[i]["faq_id"] == orig[i]["faq_id"]

    # Verify 20 PII items
    pii_items = data[100:]
    assert len(pii_items) == 20

    types = [item["pii_type"] for item in pii_items]
    assert types.count("order_id") == 5
    assert types.count("phone") == 5
    assert types.count("email") == 5
    assert types.count("upi_id") == 5

    # Verify all 20 trigger contains_pii
    for item in pii_items:
        detected, reason = contains_pii(item["question"])
        assert detected is True, f"Item {item['id']} did not trigger PII: {item['question']}"
        assert item["negative_type"] == "pii"

    # Verify split distribution: 10 dev, 10 test in PII
    splits = [item["split"] for item in pii_items]
    assert splits.count("dev") == 10
    assert splits.count("test") == 10

def test_baseline_mode_never_accesses_or_writes_to_cache():
    mock_cache = MagicMock()
    mock_embedder = MockEmbedder()
    mock_llm = MockLLM()
    mock_retriever = MockRetriever()

    # Instantiate with enable_cache=False (Baseline RAG mode)
    service = FAQService(
        cache=mock_cache,
        embedder=mock_embedder,
        llm_caller=mock_llm,
        retriever=mock_retriever,
        enable_cache=False
    )

    result = service.ask("Can I cancel my order?", virtual_clock=True)

    # 1. Cache must NEVER be searched
    assert mock_cache.search.call_count == 0

    # 2. Cache must NEVER be written to
    assert mock_cache.add.call_count == 0

    # 3. Must be marked as baseline RAG
    assert result["hit"] is False
    assert result["source"] == "llm"
    assert result["reason"] == "baseline_rag"
    assert "Mock generated answer" in result["answer"]

    # 4. LLM must have been invoked with retrieved context
    assert len(mock_llm.calls) == 1
    assert "Yes, before dispatch." in mock_llm.calls[0]["context"]

def test_baseline_mode_handles_no_retrieved_docs_gracefully():
    mock_cache = MagicMock()
    mock_embedder = MockEmbedder()
    mock_llm = MockLLM()
    mock_retriever = MockRetriever(docs=[])  # Empty retriever results

    service = FAQService(
        cache=mock_cache,
        embedder=mock_embedder,
        llm_caller=mock_llm,
        retriever=mock_retriever,
        enable_cache=False
    )

    result = service.ask("Random completely out of domain query", virtual_clock=True)

    # Cache remains untouched
    assert mock_cache.search.call_count == 0
    assert mock_cache.add.call_count == 0

    # Fallback handled correctly
    assert result["hit"] is False
    assert result["source"] == "fallback"
    assert result["reason"] == "cache_miss_no_relevant_faq"
    assert len(mock_llm.calls) == 0  # LLM should NOT be called

def test_baseline_mode_handles_pii_bypass():
    mock_cache = MagicMock()
    mock_embedder = MockEmbedder()
    mock_llm = MockLLM()
    mock_retriever = MockRetriever()

    service = FAQService(
        cache=mock_cache,
        embedder=mock_embedder,
        llm_caller=mock_llm,
        retriever=mock_retriever,
        enable_cache=False
    )

    result = service.ask("What is status of order ORD-9876543?", virtual_clock=True)

    assert mock_cache.search.call_count == 0
    assert mock_cache.add.call_count == 0
    assert result["hit"] is False
    assert "bypassed_pii" in result["reason"]
