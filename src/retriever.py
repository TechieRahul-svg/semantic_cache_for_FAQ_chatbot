import json
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


class FAQRetriever:
    """Loads FAQ documents and retrieves relevant records with FAISS."""

    def __init__(self, faqs_path: str, embedder, threshold: float = 0.45):
        with open(faqs_path, "r", encoding="utf-8") as file:
            self.documents: List[Dict[str, Any]] = json.load(file)

        self.embedder = embedder
        self.threshold = threshold
        document_texts = [self._document_text(document) for document in self.documents]
        vectors = embedder.encode(document_texts).astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.documents):
            raise ValueError("Document embeddings must match the FAQ document count.")

        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

    @staticmethod
    def _document_text(document: Dict[str, Any]) -> str:
        return "\n".join(
            [
                document.get("category", ""),
                document["question"],
                document["answer"],
            ]
        )

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index.ntotal == 0:
            return []

        vector = query_vector.reshape(1, -1).astype(np.float32)
        if vector.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension does not match document embeddings.")

        k = min(max(top_k, 1), self.index.ntotal)
        scores, indices = self.index.search(vector, k)
        return [
            (float(score), self.documents[index])
            for score, index in zip(scores[0], indices[0])
            if index >= 0 and float(score) >= self.threshold
        ]

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        return self.search(self.embedder.encode(query)[0], top_k=top_k)
