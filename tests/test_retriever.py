import json

import numpy as np

from src.cache import SemanticCache
from src.retriever import FAQRetriever
from src.service import FAQService


class FakeEmbedder:
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "return" in text.lower() else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate_answer(self, query, virtual_clock=False, context=None):
        self.calls.append({"query": query, "context": context})
        return {
            "answer": "Answer generated with FAQ context.",
            "latency_ms": 5.0,
            "total_tokens": 12,
        }


def write_faqs(tmp_path):
    path = tmp_path / "faqs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "faq_id": "faq_return",
                    "category": "Returns",
                    "question": "What is the return window?",
                    "answer": "Returns are accepted within 7 days.",
                },
                {
                    "faq_id": "faq_shipping",
                    "category": "Shipping",
                    "question": "How do I track shipping?",
                    "answer": "Use the tracking link.",
                },
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def test_faq_ingestion_builds_document_index(tmp_path):
    retriever = FAQRetriever(write_faqs(tmp_path), FakeEmbedder())

    assert len(retriever.documents) == 2
    assert retriever.index.ntotal == 2
    assert retriever.dimension == 2


def test_faq_retrieval_returns_relevant_document(tmp_path):
    retriever = FAQRetriever(write_faqs(tmp_path), FakeEmbedder(), threshold=0.8)

    results = retriever.retrieve("Please explain the return policy")

    assert len(results) == 1
    assert results[0][1]["faq_id"] == "faq_return"
    assert results[0][0] >= 0.8


def test_cache_miss_uses_retrieved_context(tmp_path):
    retriever = FAQRetriever(write_faqs(tmp_path), FakeEmbedder(), threshold=0.8)
    llm = FakeLLM()
    service = FAQService(
        cache=SemanticCache(dimension=2),
        embedder=FakeEmbedder(),
        llm_caller=llm,
        retriever=retriever,
    )

    result = service.ask("Please explain the return policy", virtual_clock=True)

    assert result["hit"] is False
    assert result["reason"] == "cache_miss_rag"
    assert result["source"] == "llm"
    assert "Returns are accepted within 7 days." in llm.calls[0]["context"]


def test_cache_miss_without_relevant_faq_uses_fallback(tmp_path):
    retriever = FAQRetriever(write_faqs(tmp_path), FakeEmbedder(), threshold=1.1)
    llm = FakeLLM()
    service = FAQService(
        cache=SemanticCache(dimension=2),
        embedder=FakeEmbedder(),
        llm_caller=llm,
        retriever=retriever,
    )

    result = service.ask("How do I track shipping?", virtual_clock=True)

    assert result["hit"] is False
    assert result["source"] == "fallback"
    assert result["reason"] == "cache_miss_no_relevant_faq"
    assert result["tokens_used"] == 0
    assert service.cache.size() == 0
    assert llm.calls == []
