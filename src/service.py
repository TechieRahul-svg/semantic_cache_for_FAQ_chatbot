import time
from typing import Dict, Any, Optional

from src.embeddings import EmbeddingModel
from src.cache import SemanticCache
from src.guards import contains_pii, is_negation_conflict
from src.llm import LLMCaller

class FAQService:
    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        embedder: Optional[EmbeddingModel] = None,
        llm_caller: Optional[LLMCaller] = None,
        threshold: float = 0.83,
        use_guard: bool = True,
        retriever=None,
        enable_cache: bool = True
    ):
        self.cache = cache
        self.embedder = embedder
        self.llm = llm_caller
        self.threshold = threshold
        self.use_guard = use_guard
        self.retriever = retriever
        self.enable_cache = enable_cache
        self.query_logs = []

    def ask(
        self,
        query: str,
        tenant_id: str = "default",
        current_time: Optional[float] = None,
        virtual_clock: bool = False
    ) -> Dict[str, Any]:
        """
        Processes a user query through the semantic caching pipeline:
        Query -> PII check -> Embed -> Lookup -> Negation guard -> Hit/Miss -> LLM -> Write
        """
        start_time = time.time()
        now = start_time if current_time is None else current_time

        # 1. Input PII Check (Scrub / bypass check)
        has_pii, pii_reason = contains_pii(query)
        if has_pii:
            # Bypass cache entirely to prevent storing or searching user-specific data
            llm_res = self.llm.generate_answer(query, virtual_clock=virtual_clock)
            total_latency = llm_res["latency_ms"] if virtual_clock else round((time.time() - start_time) * 1000.0, 2)
            
            result = {
                "answer": llm_res["answer"],
                "hit": False,
                "similarity": 0.0,
                "latency_ms": total_latency,
                "tokens_used": llm_res["total_tokens"],
                "source": "llm",
                "reason": f"bypassed_pii: {pii_reason}"
            }
            self.query_logs.append(result)
            return result

        # 2. Embed Query
        embed_start = time.time()
        query_vec = self.embedder.encode(query)[0]
        embed_ms = round((time.time() - embed_start) * 1000.0, 2)

        # 3. Cache Lookup (only when cache is enabled)
        search_results = []
        lookup_ms = 0.0
        if self.enable_cache and self.cache is not None:
            lookup_start = time.time()
            search_results = self.cache.search(
                query_vector=query_vec,
                top_k=1,
                tenant_id=tenant_id,
                current_time=now
            )
            lookup_ms = round((time.time() - lookup_start) * 1000.0, 2)

        # 4. Evaluate Hit vs Miss (only when cache is enabled)
        if self.enable_cache and search_results:
            top_sim, top_entry = search_results[0]
            if top_sim >= self.threshold:
                # 4a. Check Negation Guard
                if self.use_guard and is_negation_conflict(query, top_entry["question"]):
                    # Guard blocked: treat as miss
                    guard_blocked = True
                else:
                    guard_blocked = False

                if not guard_blocked:
                    # Cache HIT!
                    top_entry["hit_count"] = top_entry.get("hit_count", 0) + 1
                    total_latency = embed_ms + lookup_ms if virtual_clock else round((time.time() - start_time) * 1000.0, 2)

                    result = {
                        "answer": top_entry["answer"],
                        "hit": True,
                        "similarity": round(top_sim, 4),
                        "matched_faq_id": top_entry["faq_id"],
                        "matched_question": top_entry["question"],
                        "latency_ms": total_latency,
                        "tokens_used": 0,
                        "source": "cache",
                        "reason": "cache_hit"
                    }
                    self.query_logs.append(result)
                    return result

        # 5. Cache MISS or Baseline RAG -> Retrieve FAQ context, then call LLM
        retrieved_docs = self.retriever.search(query_vec, top_k=3) if self.retriever else []
        if retrieved_docs:
            context = "\n\n".join(
                f"FAQ: {document['question']}\nAnswer: {document['answer']}"
                for _, document in retrieved_docs
            )
            llm_res = self.llm.generate_answer(
                query,
                virtual_clock=virtual_clock,
                context=context
            )
            answer = llm_res["answer"]
            source = "llm"
            reason = "cache_miss_rag" if self.enable_cache else "baseline_rag"
        else:
            answer = (
                "I’m sorry, but I couldn’t find a relevant FAQ for that question. "
                "Please contact customer support for further assistance."
            )
            llm_res = {
                "latency_ms": 0.0,
                "total_tokens": 0
            }
            source = "fallback"
            reason = "cache_miss_no_relevant_faq"

        base_pipeline_ms = embed_ms + lookup_ms
        total_latency = (base_pipeline_ms + llm_res["latency_ms"]) if virtual_clock else round((time.time() - start_time) * 1000.0, 2)

        # 6. Two-Sided PII Check on Generated Answer before Cache Write (only when cache is enabled)
        answer_has_pii, ans_pii_reason = contains_pii(answer)
        if (
            self.enable_cache
            and self.cache is not None
            and source != "fallback"
            and not answer_has_pii
            and not self.cache.read_only
        ):
            # Safely write to cache
            self.cache.add(
                vector=query_vec,
                question=query,
                answer=answer,
                faq_id="dynamic_entry",
                category="dynamic",
                ttl_seconds=2592000,
                tenant_id=tenant_id,
                current_time=now
            )

        result = {
            "answer": answer,
            "hit": False,
            "similarity": round(search_results[0][0], 4) if search_results else 0.0,
            "latency_ms": total_latency,
            "tokens_used": llm_res["total_tokens"],
            "source": source,
            "reason": reason
        }
        self.query_logs.append(result)
        return result
