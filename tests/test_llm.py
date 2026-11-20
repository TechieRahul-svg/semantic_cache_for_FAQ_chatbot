import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import pytest
from src.llm import LLMCaller

def test_mock_llm_with_single_faq_context():
    caller = LLMCaller(mode="mock")
    context = (
        "FAQ: How do I cancel my order before dispatch?\n"
        "Answer: You can cancel orders before dispatch via the My Orders section."
    )
    result = caller.generate_answer("How to cancel?", virtual_clock=True, context=context)

    assert result["model"] == "mock-gpt-4o-mini"
    assert result["answer"] == "You can cancel orders before dispatch via the My Orders section."
    assert result["prompt_tokens"] > len("How to cancel?".split())
    assert result["completion_tokens"] == len(result["answer"].split())
    assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
    assert result["tokens_estimated"] is True

def test_mock_llm_with_multiple_faq_entries():
    caller = LLMCaller(mode="mock")
    context = (
        "FAQ: How do I return a product?\n"
        "Answer: You can request a return within 7 days of delivery.\n\n"
        "FAQ: How long does refund take?\n"
        "Answer: Refunds are processed within 3-5 business days."
    )
    result = caller.generate_answer("return product policy", virtual_clock=True, context=context)

    assert result["answer"] == "You can request a return within 7 days of delivery."
    assert "Refunds are processed" not in result["answer"]
    assert "FAQ:" not in result["answer"]

def test_mock_llm_empty_and_none_context():
    caller = LLMCaller(mode="mock")

    # None context
    res_none = caller.generate_answer("What is COD?", virtual_clock=True, context=None)
    assert "Thank you for contacting customer support regarding: 'What is COD?'" in res_none["answer"]

    # Empty string context
    res_empty = caller.generate_answer("What is COD?", virtual_clock=True, context="   ")
    assert "Thank you for contacting customer support regarding: 'What is COD?'" in res_empty["answer"]

def test_mock_llm_no_real_api_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    caller = LLMCaller(mode="mock")

    with patch.object(caller, "_call_real_llm") as mock_real:
        res = caller.generate_answer("Test question", virtual_clock=True)
        assert mock_real.call_count == 0
        assert res["model"] == "mock-gpt-4o-mini"

def test_mock_llm_virtual_clock_skips_sleep():
    caller = LLMCaller(mode="mock")
    start = time.time()
    caller.generate_answer("Quick test", virtual_clock=True)
    duration = time.time() - start
    assert duration < 0.2

# --- Task 3 Instrumentation Tests ---

def test_metrics_tracking_mock_only_invocation():
    caller = LLMCaller(mode="mock")
    caller.reset_metrics()

    res = caller.generate_answer("How to track?", virtual_clock=True)
    metrics = caller.get_metrics()

    assert metrics["total_generation_attempts"] == 1
    assert metrics["provider_requests"] == 0
    assert metrics["provider_successes"] == 0
    assert metrics["mock_invocations"] == 1
    assert metrics["mock_fallback_invocations"] == 0

    assert metrics["tokens"]["real_provider"]["total_tokens"] == 0
    assert metrics["tokens"]["mock_estimated"]["total_tokens_est"] == res["total_tokens"]
    assert metrics["tokens"]["mock_estimated"]["is_estimated"] is True

def test_metrics_tracking_successful_real_provider_call(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("OPENAI_API_KEY", "valid-fake-key")
    caller = LLMCaller()
    caller.reset_metrics()

    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Real API response"))],
        usage=SimpleNamespace(prompt_tokens=35, completion_tokens=15, total_tokens=50),
        model="gpt-4o-mini"
    )

    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = mock_response
    fake_openai.OpenAI.return_value = fake_client

    with patch.dict("sys.modules", {"openai": fake_openai}):
        res = caller.generate_answer("Track package", virtual_clock=True)
        assert res["tokens_estimated"] is False
        assert res["answer"] == "Real API response"

    metrics = caller.get_metrics()
    assert metrics["total_generation_attempts"] == 1
    assert metrics["provider_requests"] == 1
    assert metrics["provider_successes"] == 1
    assert metrics["mock_invocations"] == 0
    assert metrics["mock_fallback_invocations"] == 0

    assert metrics["tokens"]["real_provider"]["prompt_tokens"] == 35
    assert metrics["tokens"]["real_provider"]["completion_tokens"] == 15
    assert metrics["tokens"]["real_provider"]["total_tokens"] == 50
    assert metrics["tokens"]["mock_estimated"]["total_tokens_est"] == 0

def test_metrics_tracking_failed_real_provider_call_fallback(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("OPENAI_API_KEY", "valid-fake-key")
    caller = LLMCaller()
    caller.reset_metrics()

    context = "FAQ: Refund time\nAnswer: 3-5 days."
    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = ConnectionError("OpenAI timeout")
    fake_openai.OpenAI.return_value = fake_client

    with patch.dict("sys.modules", {"openai": fake_openai}):
        res = caller.generate_answer("When refund?", virtual_clock=True, context=context)
        assert res["answer"] == "3-5 days."
        assert res["tokens_estimated"] is True

    metrics = caller.get_metrics()
    # Verified: 1 attempt, 1 request attempted, 0 successes, 1 fallback, 0 direct mock
    assert metrics["total_generation_attempts"] == 1
    assert metrics["provider_requests"] == 1
    assert metrics["provider_successes"] == 0
    assert metrics["mock_fallback_invocations"] == 1
    assert metrics["mock_invocations"] == 0

    # Tokens charged to mock estimated, zero to provider
    assert metrics["tokens"]["real_provider"]["total_tokens"] == 0
    assert metrics["tokens"]["mock_estimated"]["total_tokens_est"] > 0

def test_metrics_reset():
    caller = LLMCaller(mode="mock")
    caller.generate_answer("Query 1", virtual_clock=True)
    caller.generate_answer("Query 2", virtual_clock=True)

    metrics_before = caller.get_metrics()
    assert metrics_before["total_generation_attempts"] == 2
    assert metrics_before["mock_invocations"] == 2

    caller.reset_metrics()
    metrics_after = caller.get_metrics()
    assert metrics_after["total_generation_attempts"] == 0
    assert metrics_after["provider_requests"] == 0
    assert metrics_after["provider_successes"] == 0
    assert metrics_after["mock_invocations"] == 0
    assert metrics_after["mock_fallback_invocations"] == 0
    assert metrics_after["tokens"]["total_tokens"] == 0

def test_multiple_invocations_without_double_counting(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    caller = LLMCaller(mode="mock")
    caller.reset_metrics()

    # Call 1: direct mock
    caller.generate_answer("Mock query 1", virtual_clock=True)
    # Call 2: direct mock
    caller.generate_answer("Mock query 2", virtual_clock=True)

    # Call 3: real failure -> fallback
    caller.mode = "real"
    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("503 Gateway Timeout")
    fake_openai.OpenAI.return_value = fake_client

    with patch.dict("sys.modules", {"openai": fake_openai}):
        caller.generate_answer("Failing query", virtual_clock=True)

    # Call 4: real success
    fake_client.chat.completions.create.side_effect = None
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Success answer"))],
        usage=SimpleNamespace(prompt_tokens=40, completion_tokens=20, total_tokens=60),
        model="gpt-4o-mini"
    )

    with patch.dict("sys.modules", {"openai": fake_openai}):
        caller.generate_answer("Successful query", virtual_clock=True)

    metrics = caller.get_metrics()
    assert metrics["total_generation_attempts"] == 4
    assert metrics["provider_requests"] == 2
    assert metrics["provider_successes"] == 1
    assert metrics["mock_invocations"] == 2
    assert metrics["mock_fallback_invocations"] == 1

    assert metrics["tokens"]["real_provider"]["prompt_tokens"] == 40
    assert metrics["tokens"]["real_provider"]["completion_tokens"] == 20
    assert metrics["tokens"]["real_provider"]["total_tokens"] == 60
    assert metrics["tokens"]["mock_estimated"]["total_tokens_est"] > 0
