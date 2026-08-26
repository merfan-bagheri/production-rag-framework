import logging
from typing import Any, Dict, List, Optional
import numpy as np

from rag_project.config import (
    FINAL_TOP_K,
    RERANKER_MODEL_NAME,
    RERANKER_FAST_MODEL,
    MIN_CHUNK_TOKEN_THRESHOLD,
    MMR_SIMILARITY_THRESHOLD,
)
from rag_project.retrieval.adaptive_strategy import AdaptiveRetrievalStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_GLOBAL_RANKERS = {}


class CrossEncoderReranker:
    """Stage 2 Ultra-High-Speed Neural Cross-Encoder Reranker using FlashRank (ms-marco-TinyBERT-L-2-v2).
    Optimized for sub-10ms pre-inference ranking, micro-chunk suppression, hardware port boosts,
    and multi-document quota balancing.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, fast_mode: bool = False):
        self.model_name = RERANKER_FAST_MODEL if fast_mode else model_name
        self.engine_type = "none"
        self.ranker = None
        self.RerankRequest = None
        self._init_model()

    def _init_model(self):
        """Initialize FlashRank Ranker or fallback."""
        global _GLOBAL_RANKERS
        if self.model_name in _GLOBAL_RANKERS:
            cached = _GLOBAL_RANKERS[self.model_name]
            self.engine_type = cached["engine_type"]
            self.ranker = cached["ranker"]
            self.RerankRequest = cached["RerankRequest"]
            return

        try:
            from flashrank import Ranker, RerankRequest
            self.engine_type = "flashrank"
            self.ranker = Ranker(model_name=self.model_name)
            self.RerankRequest = RerankRequest
            _GLOBAL_RANKERS[self.model_name] = {
                "engine_type": self.engine_type,
                "ranker": self.ranker,
                "RerankRequest": self.RerankRequest
            }
            logger.info(f"FlashRank neural reranker initialized ({self.model_name}).")
        except Exception as e:
            logger.warning(f"FlashRank init failed ({e}), trying fallback.")
            try:
                from sentence_transformers import CrossEncoder
                self.engine_type = "cross_encoder"
                self.ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                _GLOBAL_RANKERS[self.model_name] = {
                    "engine_type": self.engine_type,
                    "ranker": self.ranker,
                    "RerankRequest": None
                }
                logger.info("SentenceTransformers CrossEncoder initialized.")
            except Exception as ce_err:
                logger.warning(f"CrossEncoder init failed ({ce_err}), falling back to RRF ordering.")
                self.engine_type = "none"

    def apply_multi_doc_diversity(
        self,
        ranked_chunks: List[Dict[str, Any]],
        top_k: int,
        target_docs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Ensure balanced representation across comparative target documents."""
        if not target_docs or len(target_docs) <= 1 or not ranked_chunks:
            return ranked_chunks[:top_k]

        # Group chunks by document
        doc_pools: Dict[str, List[Dict[str, Any]]] = {d: [] for d in target_docs}
        other_pool: List[Dict[str, Any]] = []

        for c in ranked_chunks:
            doc_id = c.get("doc_id")
            if doc_id in doc_pools:
                doc_pools[doc_id].append(c)
            else:
                other_pool.append(c)

        # Allocate minimum quota per target document
        min_quota_per_doc = max(2, top_k // len(target_docs))
        balanced_selection: List[Dict[str, Any]] = []
        used_ids = set()

        # Step 1: Pick top chunks per targeted document
        for doc_id, pool in doc_pools.items():
            for c in pool[:min_quota_per_doc]:
                if c["id"] not in used_ids:
                    balanced_selection.append(c)
                    used_ids.add(c["id"])

        # Step 2: Fill remaining slots from global highest scoring candidates
        for c in ranked_chunks:
            if len(balanced_selection) >= top_k:
                break
            if c["id"] not in used_ids:
                balanced_selection.append(c)
                used_ids.add(c["id"])

        # Final sort by rerank score
        balanced_selection.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return balanced_selection[:top_k]

    def rerank(
        self,
        query: str,
        candidate_chunks: Optional[List[Dict[str, Any]]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = FINAL_TOP_K,
        hardware_port_boost: float = 0.0,
        target_docs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates against the query with micro-chunk suppression, page preservation, and multi-doc diversity."""
        chunks = candidate_chunks if candidate_chunks is not None else (candidates or [])
        if not chunks:
            return []

        # 1. Isolate page-targeted chunks and semantic candidates
        page_chunks = [c for c in chunks if c.get("page_targeted")]
        semantic_chunks = [c for c in chunks if not c.get("page_targeted")]

        # If total chunks is already small, return directly
        if len(chunks) <= top_k:
            for idx, c in enumerate(chunks):
                c["rerank_score"] = round(1.0 - (0.001 * idx), 4) if c.get("page_targeted") else c.get("rrf_score", 1.0)
            if hardware_port_boost > 0.0:
                chunks = AdaptiveRetrievalStrategy.apply_adaptive_rerank_boost(chunks, hardware_port_boost)
            return self.apply_multi_doc_diversity(chunks, top_k, target_docs)

        # 2. Score semantic candidates using FlashRank / Cross-Encoder
        reranked_semantic = []
        if semantic_chunks and self.engine_type == "flashrank" and self.ranker:
            passages = []
            for c in semantic_chunks:
                doc_title = c.get("doc_title", c.get("doc_id", "DOC"))
                sec_title = c.get("section_title", "General")
                raw_text = c.get("content", "")
                text_slice = raw_text[:800] if len(raw_text) > 800 else raw_text
                passages.append({
                    "id": c["id"],
                    "text": f"[{doc_title} | {sec_title}]\n{text_slice}",
                    "meta": c
                })

            rerank_request = self.RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_request)

            for r in results:
                chunk_data = dict(r["meta"])
                raw_score = float(r["score"])

                # Micro-chunk information density penalty (< 50 tokens)
                tok_len = chunk_data.get("token_count", len(chunk_data.get("content", "").split()))
                if tok_len < MIN_CHUNK_TOKEN_THRESHOLD and not chunk_data.get("metadata", {}).get("is_table", False):
                    penalty = (max(1, tok_len) / float(MIN_CHUNK_TOKEN_THRESHOLD)) ** 0.6
                    raw_score = raw_score * penalty

                chunk_data["rerank_score"] = round(raw_score, 6)
                reranked_semantic.append(chunk_data)

            # Re-sort after applying micro-chunk penalty
            reranked_semantic.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

            if hardware_port_boost > 0.0:
                reranked_semantic = AdaptiveRetrievalStrategy.apply_adaptive_rerank_boost(reranked_semantic, hardware_port_boost)

        elif semantic_chunks and self.engine_type == "cross_encoder" and self.ranker:
            pairs = [[query, f"[{c.get('doc_title', '')} | {c.get('section_title', '')}]\n{c.get('content', '')[:600]}"] for c in semantic_chunks]
            scores = self.ranker.predict(pairs)
            for c, s in zip(semantic_chunks, scores):
                raw_score = float(s)
                tok_len = c.get("token_count", len(c.get("content", "").split()))
                if tok_len < MIN_CHUNK_TOKEN_THRESHOLD and not c.get("metadata", {}).get("is_table", False):
                    penalty = (max(1, tok_len) / float(MIN_CHUNK_TOKEN_THRESHOLD)) ** 0.6
                    raw_score = raw_score * penalty
                c["rerank_score"] = round(raw_score, 6)
                reranked_semantic.append(c)
            reranked_semantic.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        else:
            reranked_semantic = list(semantic_chunks)

        # 3. Fuse page-targeted chunks and reranked semantic chunks
        if page_chunks:
            for idx, pc in enumerate(page_chunks):
                pc["rerank_score"] = round(0.9999 - (0.0001 * idx), 6)

            if target_docs and len(target_docs) > 1:
                page_quota = min(len(page_chunks), max(2, top_k // 2))
                selected_page = page_chunks[:page_quota]
                remaining_slots = top_k - len(selected_page)
                selected_semantic = self.apply_multi_doc_diversity(reranked_semantic, remaining_slots, target_docs)
                combined = selected_page + selected_semantic
                return combined
            else:
                combined = list(page_chunks)
                seen_ids = {c["id"] for c in page_chunks}
                for sc in reranked_semantic:
                    if len(combined) >= top_k:
                        break
                    if sc["id"] not in seen_ids:
                        combined.append(sc)
                        seen_ids.add(sc["id"])
                return combined[:top_k]

        return self.apply_multi_doc_diversity(reranked_semantic, top_k, target_docs)
