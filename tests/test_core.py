import pytest
import numpy as np
from src.guards import contains_pii, has_negation, is_negation_conflict
from src.cache import SemanticCache

def test_pii_guard_order_id_and_phone():
    # Order ID detection
    pii_found, reason = contains_pii("Where is my order ORD-9876543?")
    assert pii_found is True
    assert "order_id" in reason

    # Phone number detection
    pii_found, reason = contains_pii("Please call me on 9876543210 regarding my delivery")
    assert pii_found is True
    assert "phone" in reason

    # Clean query
    pii_found, reason = contains_pii("How do I return a damaged product?")
    assert pii_found is False
    assert reason == ""

def test_pii_guard_email_and_upi():
    # Email detection
    pii_found, reason = contains_pii("Send invoice to customer.support@gmail.com")
    assert pii_found is True
    assert "email" in reason

    # UPI detection
    pii_found, reason = contains_pii("Refund to my upi rahul@okaxis")
    assert pii_found is True
    assert "upi_id" in reason

def test_negation_detection():
    assert has_negation("Refund nahi mila abhi tak") is True
    assert has_negation("Order cancel mat karna") is True
    assert has_negation("I do not want this item") is True
    assert has_negation("Where is my package?") is False
    assert has_negation("Track my order please") is False

def test_negation_conflict():
    # Asymmetric: one has negation, other doesn't
    assert is_negation_conflict("Refund nahi chahiye", "How to request a refund?") is True
    # Symmetric: both positive
    assert is_negation_conflict("Where is my parcel?", "Track order status") is False
    # Symmetric: both negative
    assert is_negation_conflict("Refund nahi mila", "Refund has not arrived") is False

def test_virtual_clock_ttl_expiry():
    cache = SemanticCache(dimension=4)
    dummy_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Add item at virtual time = 1000.0 with TTL = 300s (expires at 1300.0)
    cache.add(
        vector=dummy_vec,
        question="What is the return window?",
        answer="7 days return window.",
        faq_id="faq_13",
        ttl_seconds=300,
        current_time=1000.0
    )

    # Search at t = 1150.0 (unexpired)
    results = cache.search(dummy_vec, top_k=1, current_time=1150.0)
    assert len(results) == 1
    assert results[0][1]["faq_id"] == "faq_13"

    # Search at t = 1301.0 (expired)
    results_expired = cache.search(dummy_vec, top_k=1, current_time=1301.0)
    assert len(results_expired) == 0

def test_tenant_isolation():
    cache = SemanticCache(dimension=4)
    dummy_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    cache.add(
        vector=dummy_vec,
        question="Tenant A policy",
        answer="Answer A",
        faq_id="faq_a",
        tenant_id="tenant_alpha",
        current_time=0.0
    )

    # Query from tenant_alpha: should find
    results_a = cache.search(dummy_vec, top_k=1, tenant_id="tenant_alpha", current_time=0.0)
    assert len(results_a) == 1
    assert results_a[0][1]["tenant_id"] == "tenant_alpha"

    # Query from tenant_beta: should be empty
    results_b = cache.search(dummy_vec, top_k=1, tenant_id="tenant_beta", current_time=0.0)
    assert len(results_b) == 0

def test_read_only_mode():
    cache = SemanticCache(dimension=4, read_only=True)
    dummy_vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    res = cache.add(
        vector=dummy_vec,
        question="Sample question",
        answer="Sample answer",
        faq_id="faq_test"
    )
    assert res == -1
    assert cache.size() == 0
