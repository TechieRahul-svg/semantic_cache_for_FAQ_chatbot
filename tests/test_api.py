import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def fast_sleep():
    with patch("time.sleep", return_value=None):
        yield

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_cache_clear_and_stats():
    res = client.delete("/cache/clear")
    assert res.status_code == 200
    assert res.json()["total_entries"] == 0

    stats = client.get("/cache/stats")
    assert stats.status_code == 200
    assert stats.json()["total_entries"] == 0

def test_ask_cold_and_hit():
    client.delete("/cache/clear")

    # 1. Cold miss
    q1 = "How do I cancel my order before it ships?"
    res1 = client.post("/ask", json={"query": q1, "tenant_id": "default"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["hit"] is False
    assert data1["source"] == "llm"

    # 2. Exact hit
    res2 = client.post("/ask", json={"query": q1, "tenant_id": "default"})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["hit"] is True
    assert data2["source"] == "cache"
    assert data2["tokens_used"] == 0
    assert data2["latency_ms"] < 200.0

def test_ask_pii_bypass():
    q = "What is the status of my order ORD-12345678?"
    res = client.post("/ask", json={"query": q})
    assert res.status_code == 200
    data = res.json()
    assert data["hit"] is False
    assert "bypassed_pii" in data["reason"]
