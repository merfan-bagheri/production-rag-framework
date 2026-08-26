import logging
import re
import time
from typing import Any, Dict, List, Optional, Union
from sentence_transformers import SentenceTransformer
import torch

from rag_project.config import (
    EMBEDDING_MODEL_NAME,
    TOP_K_DENSE,
    TOP_K_SPARSE,
    RRF_K,
    CANDIDATE_POOL_SIZE,
    DOMAIN_KEYWORDS
)
from rag_project.db.postgres_client import PostgresClient
from rag_project.retrieval.doc_router import MultiDocRouter, normalize_digits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_GLOBAL_EMBEDDER = None

def get_global_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _GLOBAL_EMBEDDER = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            _GLOBAL_EMBEDDER = SentenceTransformer(model_name, device=device)
        _GLOBAL_EMBEDDER.max_seq_length = 512
        if hasattr(_GLOBAL_EMBEDDER, "tokenizer") and _GLOBAL_EMBEDDER.tokenizer is not None:
            _GLOBAL_EMBEDDER.tokenizer.model_max_length = 512
    return _GLOBAL_EMBEDDER

class HybridSearchEngine:
    """Stage 1 Ultra-Fast Multi-Document Hybrid Retrieval:
    Dense vector cosine similarity (pgvector HNSW) + Weighted sparse lexical search (PostgreSQL tsvector)
    fused via Reciprocal Rank Fusion (RRF k=60) with deterministic page routing and sub-50ms latency.
    """

    def __init__(
        self,
        db_client: Optional[PostgresClient] = None,
        embedder: Optional[SentenceTransformer] = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        rrf_k: int = RRF_K,
    ):
        self.db = db_client or PostgresClient()
        self.rrf_k = rrf_k
        self.embedder = embedder or get_global_embedder(embedding_model_name)


    def clean_fts_query(self, query: str) -> str:
        """Sanitize query string for PostgreSQL plainto_tsquery while preserving exact acronyms."""
        norm = normalize_digits(query)
        cleaned = re.sub(r"[^\w\s_\.\-]", " ", norm)
        return " ".join(cleaned.split())

    def detect_high_specificity_acronyms(self, query: str) -> List[str]:
        """Identify critical technical tokens and acronyms requiring strong lexical matching."""
        q_lower = query.lower()
        acronyms = []
        target_tokens = DOMAIN_KEYWORDS

        for token in target_tokens:
            if token in q_lower:
                acronyms.append(token)
        return acronyms

    def search(
        self,
        query: str,
        top_k_dense: int = TOP_K_DENSE,
        top_k_sparse: int = TOP_K_SPARSE,
        top_candidates: int = CANDIDATE_POOL_SIZE,
        doc_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform high-speed hybrid search across multi-document corpus with dynamic page & intent routing."""
        # 1. Determine targeted document scope and explicit page targets
        route_info = MultiDocRouter.route_query(query)
        if doc_filter is None:
            effective_doc_filter = route_info.get("target_docs")
        else:
            effective_doc_filter = [doc_filter] if isinstance(doc_filter, str) else doc_filter

        page_targets = route_info.get("page_targets", [])

        # 2. Deterministic Page Chunk Retrieval
        page_chunks = []
        if page_targets:
            seen_page_ids = set()
            for pt in page_targets:
                target_doc = pt.get("doc_id") or (effective_doc_filter[0] if (effective_doc_filter and len(effective_doc_filter) == 1) else None)
                pages = pt.get("pages", [])
                try:
                    fetched = self.db.get_chunks_by_pages(pages, doc_id=target_doc)
                    for c in fetched:
                        if c["id"] not in seen_page_ids:
                            c["rrf_score"] = 1.0
                            c["page_targeted"] = True
                            c["requested_page"] = pt.get("requested_page")
                            page_chunks.append(c)
                            seen_page_ids.add(c["id"])
                except Exception as e:
                    logger.warning(f"Error fetching page chunks for target {pt}: {e}")

        # 3. Compute Dense Query Embedding
        query_emb = self.embedder.encode(query, batch_size=1, show_progress_bar=False, normalize_embeddings=True).tolist()
        sanitized_query = self.clean_fts_query(query)

        # Scale dense/sparse limits for multi-document scopes
        scaled_k_dense = top_k_dense + (10 if effective_doc_filter and len(effective_doc_filter) > 1 else 0)
        scaled_k_sparse = top_k_sparse + (10 if effective_doc_filter and len(effective_doc_filter) > 1 else 0)
        scaled_limit = top_candidates + (10 if effective_doc_filter and len(effective_doc_filter) > 1 else 0)

        # 4. Stratified Multi-Document or Unified Fast Path Retrieval
        try:
            if effective_doc_filter and len(effective_doc_filter) > 1:
                # Stratified retrieval: Fetch rich candidate pool from each targeted document
                candidates = []
                seen_ids = set()
                per_doc_limit = max(12, 36 // len(effective_doc_filter))
                for target_d in effective_doc_filter:
                    doc_candidates = self.db.hybrid_search_rrf(
                        query_embedding=query_emb,
                        query_text=sanitized_query,
                        top_k_dense=top_k_dense,
                        top_k_sparse=top_k_sparse,
                        rrf_k=self.rrf_k,
                        limit=per_doc_limit,
                        doc_filter=[target_d]
                    )
                    for c in doc_candidates:
                        if c["id"] not in seen_ids:
                            candidates.append(c)
                            seen_ids.add(c["id"])
            else:
                candidates = self.db.hybrid_search_rrf(
                    query_embedding=query_emb,
                    query_text=sanitized_query,
                    top_k_dense=scaled_k_dense,
                    top_k_sparse=scaled_k_sparse,
                    rrf_k=self.rrf_k,
                    limit=scaled_limit,
                    doc_filter=effective_doc_filter
                )


            if candidates:
                for c in candidates:
                    if "rrf_score" in c:
                        c["rrf_score"] = round(float(c["rrf_score"]), 6)

                # Acronym boosting
                acronyms = self.detect_high_specificity_acronyms(query)
                if acronyms:
                    for c in candidates:
                        content_lower = (str(c.get("content", "")) + " " + str(c.get("section_title", ""))).lower()
                        match_count = sum(1 for a in acronyms if a in content_lower)
                        if match_count > 0:
                            c["rrf_score"] = round(c.get("rrf_score", 0.0) + (0.08 * match_count), 6)
                            c["acronym_boost"] = 0.08 * match_count
                    candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

                # Fuse page chunks and semantic candidates
                if page_chunks:
                    seen_ids = {c["id"] for c in page_chunks}
                    combined = list(page_chunks)
                    for c in candidates:
                        if c["id"] not in seen_ids:
                            combined.append(c)
                    return combined[:max(top_candidates, len(page_chunks) + 12)]

                return candidates
        except Exception as e:
            logger.warning(f"Fast multi-doc hybrid search failed ({e})")

        return page_chunks if page_chunks else []
