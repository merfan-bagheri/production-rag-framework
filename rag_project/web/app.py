import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests

from rag_project.config import (
    BASE_DIR,
    DEFAULT_PROVIDER,
    GOOGLE_API_KEY_FILE,
    OLLAMA_BASE_URL,
    OLLAMA_PRIMARY_MODEL,
    GEMINI_DEFAULT_MODEL,
    GEMINI_FALLBACK_MODEL,
    FINAL_TOP_K,
    CANDIDATE_POOL_SIZE,
    RAG_CONFIG,
    APP_TITLE,
    APP_VERSION,
    APP_DESCRIPTION,
    SAMPLE_PROMPTS
)
from rag_project.generation.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title=APP_TITLE, version=APP_VERSION, description=APP_DESCRIPTION)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (Singleton for zero-delay instant inference)
_GLOBAL_PIPELINE: Optional[RAGPipeline] = None

def get_shared_pipeline() -> RAGPipeline:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = RAGPipeline()
    return _GLOBAL_PIPELINE

# Request & Response Schemas
class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    provider: str = Field(default="gemini")
    model: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    top_k_rerank: Optional[Union[int, str]] = None
    auto_k: bool = Field(default=False)
    fast_rerank: bool = False
    use_chat_history: bool = True
    history: List[Message] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    provider: str
    model: str
    effective_query: Optional[str] = None
    reformulated_query: Optional[str] = None
    adaptive_strategy: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]]
    timings_ms: Dict[str, Any]
    eval_tokens: int

@app.get("/api/config")
async def get_app_configuration():
    """Return runtime configuration parameters for frontend UI initialization."""
    return {
        "title": APP_TITLE,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "sample_prompts": SAMPLE_PROMPTS,
        "final_top_k": FINAL_TOP_K,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "default_provider": DEFAULT_PROVIDER,
        "default_model": GEMINI_DEFAULT_MODEL,
        "embedding_and_reranking": RAG_CONFIG.get("embedding_and_reranking", {}),
        "app": RAG_CONFIG.get("app", {})
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Process conversational query through history-aware adaptive RAG pipeline."""
    try:
        pipeline = get_shared_pipeline()

        history_payload = []
        if req.use_chat_history and req.history:
            history_payload = [{"role": m.role, "content": m.content} for m in req.history]

        result = pipeline.query(
            question=req.query,
            history=history_payload,
            top_k_rerank=req.top_k_rerank,
            auto_k=req.auto_k,
            temperature=req.temperature,
            provider=req.provider,
            model=req.model
        )


        return ChatResponse(
            answer=result["answer"],
            provider=result.get("provider", req.provider),
            model=result.get("model", req.model or ""),
            effective_query=result.get("effective_query"),
            reformulated_query=result.get("reformulated_query"),
            adaptive_strategy=result.get("adaptive_strategy"),
            sources=result.get("sources", []),
            timings_ms=result.get("timings_ms", {}),
            eval_tokens=result.get("eval_tokens", 0)
        )
    except Exception as e:
        logger.error(f"Chat API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_available_models():
    """List available models across Google AI Studio, OpenRouter, Mistral, Cohere, and Ollama."""
    models_info = {
        "google": {
            "name": "Google AI Studio",
            "available": True,
            "default": "gemini-3.5-flash-lite",
            "models": ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.7-flash"]
        },
        "openrouter": {
            "name": "OpenRouter AI",
            "available": True,
            "default": "meta-llama/llama-3.3-70b-instruct",
            "models": ["meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-chat", "anthropic/claude-3.5-sonnet"]
        },
        "mistral": {
            "name": "Mistral AI",
            "available": True,
            "default": "mistral-small-latest",
            "models": ["mistral-small-latest", "mistral-large-latest", "codestral-latest"]
        },
        "cohere": {
            "name": "Cohere AI",
            "available": True,
            "default": "command-r-08-2024",
            "models": ["command-r-08-2024", "command-r-plus-08-2024"]
        },
        "ollama": {
            "name": "Local Ollama",
            "available": False,
            "default": OLLAMA_PRIMARY_MODEL,
            "models": ["gemma3:4b", "gemma3:1b"]
        }
    }

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
        if resp.status_code == 200:
            models_info["ollama"]["available"] = True
            m_list = [m.get("name", "") for m in resp.json().get("models", [])]
            if m_list:
                models_info["ollama"]["models"] = m_list
    except Exception:
        pass

    return models_info


@app.get("/api/health")
async def health_check():
    """System health check, multi-document statistics, and database record count."""
    try:
        from rag_project.db.postgres_client import PostgresClient
        db = PostgresClient()
        chunk_count = db.get_chunk_count()
        doc_stats = db.get_doc_page_stats()
        return {
            "status": "healthy",
            "database": "connected",
            "total_documents": len(doc_stats),
            "indexed_chunks": chunk_count,
            "documents": doc_stats,
            "corpus": "Xilinx Multi-Document Production Knowledge Engine (PG036, PG058, PG065, UG380, UG389, UG682)"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database_error": str(e),
            "indexed_chunks": 0
        }


# Mount static web directory
static_dir = BASE_DIR / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page application UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Web UI is initializing...</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
