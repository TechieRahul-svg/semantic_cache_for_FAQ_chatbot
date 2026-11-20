import os
import time
import random
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMCaller:
    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or os.getenv("LLM_MODE", "mock").lower()
        self.api_key = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
        self.reset_metrics()

    def reset_metrics(self) -> None:
        """Resets all metrics and token counters."""
        self.total_generation_attempts: int = 0
        self.provider_requests: int = 0
        self.provider_successes: int = 0
        self.mock_invocations: int = 0
        self.mock_fallback_invocations: int = 0

        # Token counters (real provider vs mock estimated)
        self.provider_prompt_tokens: int = 0
        self.provider_completion_tokens: int = 0

        self.mock_prompt_tokens_est: int = 0
        self.mock_completion_tokens_est: int = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Returns a structured snapshot of invocation and token metrics."""
        return {
            "total_generation_attempts": self.total_generation_attempts,
            "provider_requests": self.provider_requests,
            "provider_successes": self.provider_successes,
            "mock_invocations": self.mock_invocations,
            "mock_fallback_invocations": self.mock_fallback_invocations,
            "tokens": {
                "real_provider": {
                    "prompt_tokens": self.provider_prompt_tokens,
                    "completion_tokens": self.provider_completion_tokens,
                    "total_tokens": self.provider_prompt_tokens + self.provider_completion_tokens,
                    "is_estimated": False
                },
                "mock_estimated": {
                    "prompt_tokens_est": self.mock_prompt_tokens_est,
                    "completion_tokens_est": self.mock_completion_tokens_est,
                    "total_tokens_est": self.mock_prompt_tokens_est + self.mock_completion_tokens_est,
                    "is_estimated": True
                },
                "total_tokens": (
                    self.provider_prompt_tokens
                    + self.provider_completion_tokens
                    + self.mock_prompt_tokens_est
                    + self.mock_completion_tokens_est
                )
            }
        }

    def generate_answer(
        self,
        query: str,
        virtual_clock: bool = False,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer using either live LLM API or calibrated mock backend.
        Returns:
            {
                "answer": str,
                "latency_ms": float,
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
                "tokens_estimated": bool,
                "model": str
            }
        """
        self.total_generation_attempts += 1

        if self.mode == "real" and self.api_key:
            return self._call_real_llm(query, context=context, virtual_clock=virtual_clock)
        else:
            return self._call_mock_llm(query, virtual_clock=virtual_clock, context=context, is_fallback=False)

    def _extract_mock_answer(self, query: str, context: Optional[str]) -> str:
        """
        Extracts grounded answer from retrieved FAQ context deterministically.
        Safely parses the first complete FAQ entry and extracts its answer without
        bleeding into subsequent FAQ entries.
        """
        if not context or not context.strip():
            return (
                f"Thank you for contacting customer support regarding: '{query}'. "
                "Our automated policy team has resolved your inquiry. Please refer to your account details for updates."
            )

        text = context.strip()

        # Look for "Answer:" delimiter
        match = re.search(r"Answer:\s*(.*?)(?=\n+FAQ:|\Z)", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted

        # If no "Answer:" delimiter found, return first paragraph/block
        blocks = text.split("\n\n")
        return blocks[0].strip()

    def _call_mock_llm(
        self,
        query: str,
        virtual_clock: bool,
        context: Optional[str] = None,
        is_fallback: bool = False
    ) -> Dict[str, Any]:
        if not is_fallback:
            self.mock_invocations += 1

        # Calibrated realistic LLM response latency: Normal(1100ms, 180ms), clipped [750ms, 1800ms]
        simulated_latency_sec = max(0.75, min(1.8, random.gauss(1.1, 0.18)))
        simulated_latency_ms = round(simulated_latency_sec * 1000.0, 2)

        if not virtual_clock:
            time.sleep(simulated_latency_sec)

        answer = self._extract_mock_answer(query, context)

        # Approximate token estimates: prompt overhead + query + context words
        context_words = len(context.split()) if context else 0
        prompt_tokens_est = 25 + len(query.split()) + context_words
        completion_tokens_est = max(1, len(answer.split()))
        total_tokens_est = prompt_tokens_est + completion_tokens_est

        self.mock_prompt_tokens_est += prompt_tokens_est
        self.mock_completion_tokens_est += completion_tokens_est

        return {
            "answer": answer,
            "latency_ms": simulated_latency_ms,
            "prompt_tokens": prompt_tokens_est,
            "completion_tokens": completion_tokens_est,
            "total_tokens": total_tokens_est,
            "tokens_estimated": True,
            "model": "mock-gpt-4o-mini"
        }

    def _call_real_llm(
        self,
        query: str,
        context: Optional[str] = None,
        virtual_clock: bool = False
    ) -> Dict[str, Any]:
        """
        Optional live API caller for demo verification.
        Supports standard OpenAI-compatible endpoints or Google Gemini.
        """
        start_time = time.time()
        self.provider_requests += 1

        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a concise e-commerce support assistant."},
                        {
                            "role": "user",
                            "content": (
                                f"Use the following FAQ context to answer the question. "
                                f"If it does not answer the question, say so clearly.\n\n"
                                f"FAQ context:\n{context}\n\nQuestion: {query}"
                                if context
                                else query
                            )
                        }
                    ],
                    max_tokens=100
                )
                latency_ms = round((time.time() - start_time) * 1000.0, 2)
                usage = response.usage

                self.provider_successes += 1
                self.provider_prompt_tokens += usage.prompt_tokens
                self.provider_completion_tokens += usage.completion_tokens

                return {
                    "answer": response.choices[0].message.content,
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "tokens_estimated": False,
                    "model": response.model
                }
            else:
                raise ValueError("No OpenAI API key found for real LLM request attempt")
        except Exception as e:
            print(f"[Real LLM Fallback to Mock due to error: {e}]")
            self.mock_fallback_invocations += 1
        
        # Fallback to calibrated mock if live call fails
        return self._call_mock_llm(query, virtual_clock=virtual_clock, context=context, is_fallback=True)
