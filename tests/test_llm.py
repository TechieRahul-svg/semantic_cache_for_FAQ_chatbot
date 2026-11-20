import time
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

def test_mock_llm_with_multiple_faq_entries():
    caller = LLMCaller(mode="mock")
    context = (
        "FAQ: How do I return a product?\n"
        "Answer: You can request a return within 7 days of delivery.\n\n"
        "FAQ: How long does refund take?\n"
        "Answer: Refunds are processed within 3-5 business days."
    )
    result = caller.generate_answer("return product policy", virtual_clock=True, context=context)

    # Should extract only the first complete FAQ answer deterministically
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
    assert duration < 0.2  # Should return immediately without sleeping 1s+

def test_real_llm_fallback_to_mock_preserves_context(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "real")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    caller = LLMCaller()

    context = "FAQ: What is your return policy?\nAnswer: 7 days free returns."

    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("API failure")
    fake_openai.OpenAI.return_value = fake_client

    with patch.dict("sys.modules", {"openai": fake_openai}):
        res = caller.generate_answer("return query", virtual_clock=True, context=context)
        assert res["model"] == "mock-gpt-4o-mini"
        assert res["answer"] == "7 days free returns."
