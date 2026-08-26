import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rag_project.config import (
    DEFAULT_PROVIDER,
    GEMINI_DEFAULT_MODEL,
    OLLAMA_PRIMARY_MODEL,
    FINAL_TOP_K,
    CANDIDATE_POOL_SIZE,
    TOP_K_DENSE,
    TOP_K_SPARSE,
    GOOGLE_API_KEY_FILE,
    NEIGHBOR_EXPANSION_ENABLED,
    NEIGHBOR_EXPANSION_MIN_SCORE,
    NEIGHBOR_EXPANSION_MAX_CHUNKS
)
from rag_project.retrieval.hybrid_search import HybridSearchEngine
from rag_project.retrieval.reranker import CrossEncoderReranker
from rag_project.retrieval.query_reformulator import QueryReformulator
from rag_project.retrieval.adaptive_strategy import AdaptiveRetrievalStrategy
from rag_project.retrieval.doc_router import MultiDocRouter
from rag_project.retrieval.neighbor_expander import NeighborExpander
from rag_project.retrieval.overlap_merger import OverlapMerger
from rag_project.generation.gemini_client import GeminiClient
from rag_project.generation.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Production-Grade Multi-Document RAG Orchestration Pipeline."""

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: Optional[str] = None,
        api_key_file: Path = GOOGLE_API_KEY_FILE,
        fast_rerank: bool = False,
    ):
        self.provider = provider
        self.model_name = model or (GEMINI_DEFAULT_MODEL if provider == "gemini" else OLLAMA_PRIMARY_MODEL)

        logger.info("Initializing Hybrid Search Engine...")
        self.hybrid_engine = HybridSearchEngine()

        logger.info("Initializing Cross-Encoder Neural Reranker...")
        self.reranker = CrossEncoderReranker(fast_mode=fast_rerank)

        logger.info("Initializing Query Reformulation & Adaptive Auto-K Strategy...")
        self.reformulator = QueryReformulator()
        self.adaptive_strategy = AdaptiveRetrievalStrategy()

        logger.info("Initializing Dynamic Neighbor Lookahead Expander...")
        self.neighbor_expander = NeighborExpander(
            min_score=NEIGHBOR_EXPANSION_MIN_SCORE,
            max_expansions=NEIGHBOR_EXPANSION_MAX_CHUNKS
        )

        logger.info("Initializing Exact Overlap Reducer & Passage Stitcher...")
        self.overlap_merger = OverlapMerger()

        logger.info(f"Initializing LLM Engine (Provider: {self.provider.upper()}, Model: {self.model_name})...")
        if self.provider in ["gemini", "multi", "auto", "mistral", "cohere", "openrouter", "google"]:
            try:
                from rag_project.generation.multi_provider_engine import MultiProviderEngine
                self.llm_client = MultiProviderEngine(preferred_provider=self.provider, preferred_model=self.model_name)
            except Exception as e:
                logger.warning(f"MultiProviderEngine init fallback to GeminiClient: {e}")
                self.llm_client = GeminiClient(api_key_file=api_key_file, model_name=self.model_name)
        else:
            self.llm_client = OllamaClient(model_name=self.model_name)



        # Pre-warm models to eliminate cold-start latency spikes
        try:
            self.hybrid_engine.embedder.encode(["warmup query"], normalize_embeddings=True)
            from rag_project.retrieval.overlap_merger import get_tokenizer
            get_tokenizer()
            logger.info("RAG Pipeline pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"Pipeline warm-up notice: {e}")

    def query(
        self,
        query_text: Optional[str] = None,
        question: Optional[str] = None,
        query: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        top_k_rerank: Optional[Union[int, str]] = None,
        auto_k: bool = False,
        doc_filter: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.1,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end multi-document RAG query with pre-inference timing."""
        effective_query = query_text or question or query or ""
        if not effective_query:
            raise ValueError("Query string cannot be empty.")

        overall_start = time.time()
        timings = {}

        # 1. Multi-Turn Query Reformulation
        reform_res = self.reformulator.reformulate(effective_query, history or [])
        retrieval_query = reform_res["reformulated_query"]
        timings["reformulation"] = reform_res["latency_ms"]

        # 2. Document Scope Routing
        route_info = MultiDocRouter.route_query(retrieval_query)
        effective_doc_filter = doc_filter or route_info.get("target_docs")

        # 3. Determine Effective K
        req_k = int(top_k_rerank) if (top_k_rerank is not None and str(top_k_rerank).isdigit()) else FINAL_TOP_K
        if auto_k:
            adaptive_cfg = self.adaptive_strategy.analyze_and_adapt(retrieval_query, base_k=req_k)
            effective_k = adaptive_cfg.get("effective_k", req_k)
            port_boost = adaptive_cfg.get("port_boost", 0.0)
        else:
            adaptive_cfg = {
                "effective_k": req_k,
                "port_boost": 0.0,
                "strategy_mode": "manual",
                "rationale": f"Manual top_k mode (k={req_k})."
            }
            effective_k = req_k
            port_boost = 0.0

        # 4. Stage 1: Multi-Doc Hybrid Retrieval
        t_ret_start = time.time()
        pool_size = max(CANDIDATE_POOL_SIZE, effective_k * 2, 35)
        candidates = self.hybrid_engine.search(
            retrieval_query,
            top_candidates=pool_size,
            top_k_dense=max(TOP_K_DENSE, effective_k * 2),
            top_k_sparse=max(TOP_K_SPARSE, effective_k * 2),
            doc_filter=effective_doc_filter
        )
        timings["hybrid_retrieval"] = round((time.time() - t_ret_start) * 1000, 2)

        # 5. Stage 2: Cross-Encoder Neural Reranking with Multi-Doc Quota Balancing
        t_rerank_start = time.time()
        reranked_chunks = self.reranker.rerank(
            query=retrieval_query,
            candidates=candidates,
            top_k=effective_k,
            hardware_port_boost=port_boost,
            target_docs=effective_doc_filter
        )
        timings["reranking"] = round((time.time() - t_rerank_start) * 1000, 2)

        # 6. Stage 2.2: Dynamic Neighbor Lookahead Expander
        t_expand_start = time.time()
        expanded_chunks = reranked_chunks
        if NEIGHBOR_EXPANSION_ENABLED:
            expanded_chunks = self.neighbor_expander.expand_neighbors(
                ranked_chunks=reranked_chunks,
                target_docs=effective_doc_filter
            )
        timings["neighbor_expansion"] = round((time.time() - t_expand_start) * 1000, 2)

        # 7. Stage 2.5: Exact Overlap Reducer & Passage Stitcher
        t_dedup_start = time.time()
        dedup_res = self.overlap_merger.reduce_chunks(expanded_chunks)
        effective_context = dedup_res.get("reduced_chunks", expanded_chunks)
        timings["overlap_deduplication"] = round((time.time() - t_dedup_start) * 1000, 2)

        # Pre-Inference Total Latency
        pre_inf_total = round(
            timings["reformulation"] + timings["hybrid_retrieval"] + timings["reranking"] + timings["neighbor_expansion"] + timings["overlap_deduplication"],
            2
        )
        timings["pre_inference_total"] = pre_inf_total

        # 7. Stage 3: LLM Generation
        t_gen_start = time.time()
        try:
            llm_out = self.llm_client.generate(
                query=effective_query,
                context_chunks=effective_context,
                temperature=temperature,
                preferred_provider=provider,
                preferred_model=model
            )
        except TypeError:
            llm_out = self.llm_client.generate(
                query=effective_query,
                context_chunks=effective_context,
                temperature=temperature
            )
        timings["generation"] = round((time.time() - t_gen_start) * 1000, 2)
        timings["total"] = round((time.time() - overall_start) * 1000, 2)

        return {
            "query": effective_query,
            "effective_query": effective_query,
            "reformulated_query": retrieval_query,
            "retrieval_query": retrieval_query,
            "route_info": route_info,
            "adaptive_config": adaptive_cfg,
            "answer": llm_out.get("response") or llm_out.get("text") or "",
            "response": llm_out.get("response") or llm_out.get("text") or "",
            "model": llm_out.get("model", self.model_name),
            "model_used": llm_out.get("model", self.model_name),
            "provider": llm_out.get("provider", self.provider),
            "sources": effective_context,
            "raw_sources": reranked_chunks,
            "deduplication_metrics": {
                "original_tokens": dedup_res.get("original_tokens", 0),
                "reduced_tokens": dedup_res.get("reduced_tokens", 0),
                "tokens_saved": dedup_res.get("tokens_saved", 0),
                "reduction_pct": dedup_res.get("reduction_pct", 0.0),
                "false_deletion_count": dedup_res.get("false_deletion_count", 0),
                "information_coverage_pct": dedup_res.get("information_coverage_pct", 100.0),
            },
            "timings_ms": timings,
            "eval_tokens": llm_out.get("eval_count", 0),
        }

