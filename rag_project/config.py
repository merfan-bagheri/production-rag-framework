import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "doc.pdf"
DOCS_DIR = PROJECT_ROOT / "docs"

# Configuration File Paths
CONFIG_FILE = PROJECT_ROOT / "rag_config.json"
CONFIG_EXAMPLE_FILE = PROJECT_ROOT / "rag_config.example.json"
GOOGLE_API_KEY_FILE = PROJECT_ROOT / "google-api-key.txt"
APIS_FILE = PROJECT_ROOT / "APIs.txt"
APIS_EXAMPLE_FILE = PROJECT_ROOT / "APIs.example.txt"
SYSTEM_PROMPT_FILE = PROJECT_ROOT / "system_prompt.txt"
SYSTEM_PROMPT_EXAMPLE_FILE = PROJECT_ROOT / "system_prompt.example.txt"


def load_json_config() -> Dict[str, Any]:
    """Load master RAG configuration from rag_config.json or fallback to rag_config.example.json."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading {CONFIG_FILE}: {e}. Falling back to default example config.")

    if CONFIG_EXAMPLE_FILE.exists():
        try:
            with open(CONFIG_EXAMPLE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading {CONFIG_EXAMPLE_FILE}: {e}")

    return {}


# Load raw JSON configuration dictionary
RAG_CONFIG: Dict[str, Any] = load_json_config()

# Helper Accessor Functions
def get_app_config() -> Dict[str, Any]:
    return RAG_CONFIG.get("app", {})

def get_database_config() -> Dict[str, Any]:
    return RAG_CONFIG.get("database", {})

def get_embedding_and_reranking_config() -> Dict[str, Any]:
    return RAG_CONFIG.get("embedding_and_reranking", {})

def get_llm_providers_config() -> Dict[str, Any]:
    return RAG_CONFIG.get("llm_providers", {})

def get_retrieval_and_routing_config() -> Dict[str, Any]:
    return RAG_CONFIG.get("retrieval_and_routing", {})

def get_ingestion_and_chunking_config() -> Dict[str, Any]:
    return RAG_CONFIG.get("ingestion_and_chunking", {})

def get_domain_keywords() -> List[str]:
    return get_retrieval_and_routing_config().get("domain_keywords", [])

def get_document_registry() -> Dict[str, Any]:
    return get_retrieval_and_routing_config().get("document_registry", {})

def get_comparison_patterns() -> List[str]:
    return get_retrieval_and_routing_config().get("comparison_patterns", [])

def get_adaptive_intent_strategies() -> Dict[str, Any]:
    return get_retrieval_and_routing_config().get("adaptive_intent_strategies", {})

def get_query_reformulation_config() -> Dict[str, Any]:
    return get_retrieval_and_routing_config().get("query_reformulation", {})

def get_header_footer_patterns() -> List[str]:
    return get_ingestion_and_chunking_config().get("header_footer_strip_patterns", [])

def get_document_metadata_map() -> Dict[str, Any]:
    return get_ingestion_and_chunking_config().get("document_metadata_map", {})


# System Prompt Loader
DEFAULT_SYSTEM_PROMPT = """You are an autonomous Principal Systems Engineer acting as the authoritative technical architect across the ingested technical documentation.

CRITICAL OPERATIONAL & REASONING PROTOCOL:
1. ZERO-HALLUCINATION & FACTUAL PURITY:
   - Rely EXCLUSIVELY on the provided Retrieved Context below. Do NOT assume, extrapolate, or inject external undocumented claims.
   - When answering comparative queries across multiple documents/systems, clearly isolate the architectural features, primitive types, and specifications of each subsystem without mixing parameters across tables.
   - If an architecture, feature, or interface is NOT documented in the retrieved context, explicitly state: "The provided technical documentation context does not contain information regarding [specific entity/feature]."

2. MULTI-DOCUMENT CITATION DENSITY:
   - Every individual factual statement, parameter definition, timing latency specification, or system constraint MUST include an explicit citation in the format:
     [Doc: <doc_title> | Page: <page_number> | Section: <section_breadcrumb>]
   - Do NOT produce uncited technical claims.

3. ARCHITECTURAL & COMPARATIVE PRECISION:
   - Accurately distinguish primitive and subsystem implementations.
   - State exact signal directions, clock domains, operational modes, and register latencies.
   - Use Markdown tables and bold formatting for structural clarity.

4. PAGE & LOCALIZED SECTION REASONING:
   - When the user asks about a specific page number or localized section (e.g. 'what is on page 30?', 'what about section 3.2?'):
     * Inspect the `[Doc: <doc_title> | Page: <page_number> | Section: <breadcrumb>]` metadata in the chunk headers.
     * Detail all functional use models, timing diagrams, signal tables, formulas, and configurations belonging to the requested page chunks thoroughly.
     * Always cite the exact page and section in each statement.

5. EXHAUSTIVE TECHNICAL COMPLETENESS:
   - Detail exact entity names, registers, bitfields, pinouts, command-line arguments, and mathematical equations.
   - Answer all sub-parts of the query exhaustively with zero omission."""


def load_system_prompt(file_path: Optional[Path] = None) -> str:
    """Dynamically load system prompt from file (defaults to system_prompt.txt or system_prompt.example.txt).
    Falls back to DEFAULT_SYSTEM_PROMPT if file does not exist.
    """
    target = file_path or SYSTEM_PROMPT_FILE
    if target and target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass

    if SYSTEM_PROMPT_EXAMPLE_FILE.exists():
        try:
            with open(SYSTEM_PROMPT_EXAMPLE_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass

    return DEFAULT_SYSTEM_PROMPT


# App Configuration Constants
_app_cfg = get_app_config()
APP_TITLE = _app_cfg.get("title", "Xilinx LogiCORE IP RAG Expert")
APP_VERSION = _app_cfg.get("version", "2.5.0")
APP_DESCRIPTION = _app_cfg.get("description", "High-Precision Production RAG Engine")
APP_DEFAULT_PORT = int(_app_cfg.get("default_port", 8816))
SAMPLE_PROMPTS = _app_cfg.get("sample_prompts", [])

# Database Configuration (PostgreSQL + pgvector)
_db_cfg = get_database_config()
DB_HOST = os.getenv("POSTGRES_HOST", _db_cfg.get("host", "localhost"))
DB_PORT = int(os.getenv("POSTGRES_PORT", _db_cfg.get("port", 15432)))
DB_FALLBACK_PORT = int(_db_cfg.get("fallback_port", 5432))
DB_NAME = os.getenv("POSTGRES_DB", _db_cfg.get("name", "rag_db"))
DB_USER = os.getenv("POSTGRES_USER", _db_cfg.get("user", "postgres"))
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", _db_cfg.get("password", "postgres"))

# Docker Configuration
DOCKER_CONTAINER_NAME = _db_cfg.get("container_name", "rag_postgres")
DOCKER_IMAGE = _db_cfg.get("docker_image", "pgvector/pgvector:pg16")
DOCKER_VOLUME_NAME = _db_cfg.get("docker_volume", "rag_pgvector_data")

# Embedding & Reranker Model Configuration
_model_cfg = get_embedding_and_reranking_config()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", _model_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
EMBEDDING_DIM = int(_model_cfg.get("embedding_dim", 384))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", _model_cfg.get("reranker_model", "ms-marco-TinyBERT-L-2-v2"))
RERANKER_FAST_MODEL = _model_cfg.get("reranker_fast_model", "ms-marco-TinyBERT-L-2-v2")
CANDIDATE_POOL_SIZE = int(_model_cfg.get("candidate_pool_size", 60))

# Retrieval & Fusion Parameters
TOP_K_DENSE = int(_model_cfg.get("top_k_dense", 60))
TOP_K_SPARSE = int(_model_cfg.get("top_k_sparse", 60))
RRF_K = int(_model_cfg.get("rrf_k", 60))
FINAL_TOP_K = int(_model_cfg.get("final_top_k", 25))

# LLM Providers Configuration
_llm_cfg = get_llm_providers_config()
_providers_map = _llm_cfg.get("providers", {})
DEFAULT_PROVIDER = os.getenv("RAG_PROVIDER", _llm_cfg.get("default_provider", "gemini"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", _llm_cfg.get("max_output_tokens", 16384)))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", _llm_cfg.get("timeout_seconds", 90)))

# Ollama LLM Configuration
_ollama_cfg = _providers_map.get("ollama", {})
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", _ollama_cfg.get("base_url", "http://localhost:11434"))
OLLAMA_PRIMARY_MODEL = os.getenv("OLLAMA_MODEL", _ollama_cfg.get("default_model", "gemma3:4b"))
OLLAMA_FALLBACK_MODEL = _ollama_cfg.get("fallback_model", "gemma3:1b")

# Google AI Studio (Gemini) Configuration
_google_cfg = _providers_map.get("google", {})
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", _google_cfg.get("default_model", "gemini-3.5-flash-lite"))
GEMINI_FALLBACK_MODEL = _google_cfg.get("fallback_model", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

# Chunking & Ingestion Configuration
_ingest_cfg = get_ingestion_and_chunking_config()
MIN_CHUNK_TOKENS = int(_ingest_cfg.get("min_chunk_tokens", 75))
CHUNK_TARGET_TOKENS = int(_ingest_cfg.get("chunk_target_tokens", 400))
CHUNK_OVERLAP_TOKENS = int(_ingest_cfg.get("chunk_overlap_tokens", 80))
MAX_CHUNK_CHARS = int(_ingest_cfg.get("max_chunk_chars", 1800))
OVERLAP_CHARS = int(_ingest_cfg.get("overlap_chars", 200))
IGNORE_SECTIONS = _ingest_cfg.get("ignore_sections", ["Revision History", "Notice of Disclaimer", "Table of Contents", "Conventions"])

# Post-Rerank & Deduplication Parameters
MIN_CHUNK_TOKEN_THRESHOLD = int(_model_cfg.get("min_chunk_token_threshold", 75))
MMR_SIMILARITY_THRESHOLD = float(_model_cfg.get("mmr_similarity_threshold", 0.85))
NEIGHBOR_EXPANSION_ENABLED = bool(_model_cfg.get("neighbor_expansion_enabled", True))
NEIGHBOR_EXPANSION_MIN_SCORE = float(_model_cfg.get("neighbor_expansion_min_score", 0.05))
NEIGHBOR_EXPANSION_MAX_CHUNKS = int(_model_cfg.get("neighbor_expansion_max_chunks", 12))

# Exported domain-specific definitions (Loaded from rag_config.json)
DOMAIN_KEYWORDS = get_domain_keywords()
DOCUMENT_REGISTRY = get_document_registry()
COMPARISON_PATTERNS = get_comparison_patterns()
ADAPTIVE_INTENT_STRATEGIES = get_adaptive_intent_strategies()
QUERY_REFORMULATION_CONFIG = get_query_reformulation_config()
HEADER_FOOTER_PATTERNS = get_header_footer_patterns()
DOCUMENT_METADATA_MAP = get_document_metadata_map()
