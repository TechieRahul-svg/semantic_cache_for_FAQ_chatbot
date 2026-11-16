import os
import sys
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.embeddings import EmbeddingModel
from src.cache import SemanticCache
from src.llm import LLMCaller
from src.service import FAQService

app = FastAPI(
    title="FAQ Semantic Cache Service",
    description="High-performance semantic cache with FAISS, negation guardrails, and two-sided PII scrubbing.",
    version="1.0.0"
)

# Global service instance
embedder = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
cache = SemanticCache(dimension=384, read_only=False)
llm = LLMCaller()
service = FAQService(cache=cache, embedder=embedder, llm_caller=llm, threshold=0.83, use_guard=True)

class AskRequest(BaseModel):
    query: str = Field(..., example="How do I cancel my order before dispatch?")
    tenant_id: str = Field(default="default", example="default")

class AskResponse(BaseModel):
    answer: str
    hit: bool
    similarity: float
    latency_ms: float
    tokens_used: int
    source: str
    reason: str
    matched_faq_id: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "ok", "cached_entries": cache.size()}

@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = service.ask(
        query=req.query,
        tenant_id=req.tenant_id,
        virtual_clock=False
    )
    return result

@app.get("/cache/stats")
def get_cache_stats():
    total_entries = cache.size()
    total_hits = sum(e.get("hit_count", 0) for e in cache.entries)
    tenants = list(set(e.get("tenant_id", "default") for e in cache.entries))
    return {
        "total_entries": total_entries,
        "total_cache_hits": total_hits,
        "active_tenants": tenants,
        "threshold": service.threshold,
        "negation_guard_enabled": service.use_guard
    }

@app.post("/cache/prewarm")
def prewarm_cache():
    faqs_path = os.path.join(os.path.dirname(__file__), "..", "data", "faqs_30.json")
    if not os.path.exists(faqs_path):
        raise HTTPException(status_code=404, detail="faqs_30.json not found")
    cache.warm_from_faqs(faqs_path, embedder)
    return {"status": "success", "total_entries": cache.size()}

@app.delete("/cache/clear")
def clear_cache():
    global cache, service
    cache = SemanticCache(dimension=384, read_only=False)
    service.cache = cache
    return {"status": "cleared", "total_entries": 0}
