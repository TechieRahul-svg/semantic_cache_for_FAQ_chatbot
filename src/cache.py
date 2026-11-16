import time
import json
import numpy as np
import faiss
from typing import Optional, Dict, Any, List, Tuple

class SemanticCache:
    def __init__(self, dimension: int = 384, read_only: bool = False):
        self.dimension = dimension
        self.read_only = read_only
        self.index = faiss.IndexFlatIP(dimension)
        self.entries: List[Dict[str, Any]] = []

    def set_read_only(self, read_only: bool):
        self.read_only = read_only

    def size(self) -> int:
        return self.index.ntotal

    def add(
        self,
        vector: np.ndarray,
        question: str,
        answer: str,
        faq_id: str,
        category: str = "General",
        ttl_seconds: int = 2592000,
        tenant_id: str = "default",
        current_time: Optional[float] = None
    ) -> int:
        """
        Adds a new question-answer entry to FAISS and the metadata store.
        If cache is in read_only mode, writes are ignored.
        """
        if self.read_only:
            return -1

        now = time.time() if current_time is None else current_time
        vec = vector.reshape(1, -1).astype(np.float32)

        entry_idx = len(self.entries)
        entry = {
            "id": entry_idx,
            "faq_id": faq_id,
            "question": question,
            "answer": answer,
            "category": category,
            "ttl_seconds": ttl_seconds,
            "tenant_id": tenant_id,
            "timestamp": now,
            "hit_count": 0
        }

        self.index.add(vec)
        self.entries.append(entry)
        return entry_idx

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        tenant_id: str = "default",
        current_time: Optional[float] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Searches FAISS for the nearest entries matching tenant_id and unexpired TTL.
        Returns a list of (similarity_score, entry) sorted descending.
        """
        if self.index.ntotal == 0:
            return []

        now = time.time() if current_time is None else current_time
        q_vec = query_vector.reshape(1, -1).astype(np.float32)
        
        # Search more candidates in case some belong to another tenant or are expired
        k = min(top_k * 2, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        valid_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            
            # Tenant isolation
            if entry.get("tenant_id", "default") != tenant_id:
                continue
            
            # TTL Expiry check via virtual clock
            ttl = entry.get("ttl_seconds", 2592000)
            if now > entry["timestamp"] + ttl:
                continue

            valid_results.append((float(score), entry))
            if len(valid_results) >= top_k:
                break

        return valid_results

    def warm_from_faqs(
        self,
        faqs_path: str,
        embedder,
        tenant_id: str = "default",
        current_time: float = 0.0
    ):
        """
        Pre-warms the cache with canonical FAQs.
        """
        with open(faqs_path, "r", encoding="utf-8") as f:
            faqs = json.load(f)

        questions = [f["question"] for f in faqs]
        vectors = embedder.encode(questions)

        for faq, vec in zip(faqs, vectors):
            self.add(
                vector=vec,
                question=faq["question"],
                answer=faq["answer"],
                faq_id=faq["faq_id"],
                category=faq.get("category", "General"),
                ttl_seconds=faq.get("ttl_seconds", 2592000),
                tenant_id=tenant_id,
                current_time=current_time
            )
