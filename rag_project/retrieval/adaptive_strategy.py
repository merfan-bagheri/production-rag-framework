import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rag_project.config import ADAPTIVE_INTENT_STRATEGIES, FINAL_TOP_K

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AdaptiveRetrievalStrategy:
    """Intelligent hardware-aware retrieval strategy optimizer.
    Dynamically tunes Top-K chunk count and applies structural priority weights based on query intent.
    """

    @classmethod
    def analyze_and_adapt(cls, query: str, base_k: Optional[int] = None) -> Dict[str, Any]:
        """Unified method to analyze query intent and return effective parameters."""
        info = cls.analyze_query_intent(query, base_k=base_k)
        return {
            "effective_k": info.get("recommended_k", base_k or FINAL_TOP_K),
            "port_boost": info.get("port_boost", 0.0),
            "strategy_mode": info.get("strategy_mode", "standard"),
            "rationale": info.get("rationale", "")
        }

    @staticmethod
    def analyze_query_intent(query: str, base_k: Optional[int] = None) -> Dict[str, Any]:
        """Analyze query intent to determine optimal chunk count (k) and rerank weight boosts."""
        q = query.lower()
        strategies = ADAPTIVE_INTENT_STRATEGIES
        target_base = int(base_k) if base_k is not None else FINAL_TOP_K

        for mode_name, strat in strategies.items():
            signals = strat.get("signals", [])
            if signals and any(kw in q for kw in signals):
                rec_k = max(strat.get("recommended_k", 7), target_base)
                return {
                    "strategy_mode": mode_name,
                    "recommended_k": rec_k,
                    "port_boost": strat.get("port_boost", 0.0),
                    "rationale": strat.get("rationale", f"Triggered intent mode: {mode_name}")
                }

        # Fallback mode
        fallback = strategies.get("focused_lookup", {})
        rec_k = max(fallback.get("recommended_k", 6), target_base)
        return {
            "strategy_mode": "focused_lookup",
            "recommended_k": rec_k,
            "port_boost": fallback.get("port_boost", 0.0),
            "rationale": fallback.get("rationale", f"Standard focused query (k={rec_k}).")
        }

    @staticmethod
    def apply_adaptive_rerank_boost(
        chunks: List[Dict[str, Any]],
        port_boost: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Apply structural weight boost for key domain entities and table chunks."""
        if port_boost <= 0.0 or not chunks:
            return chunks

        from rag_project.config import DOMAIN_KEYWORDS
        domain_kws = DOMAIN_KEYWORDS or ["table", "specification", "parameter", "register", "signal"]

        boosted_chunks = []
        for c in chunks:
            item = dict(c)
            base_score = float(item.get("rerank_score", item.get("rrf_score", 0.0)))
            breadcrumb = str(item.get("breadcrumb", "")).lower()
            content = str(item.get("content", "")).lower()
            content_type = str(item.get("content_type", "")).lower()

            is_structural_match = (
                content_type in ["atomic_table", "table", "spec"] or
                "table" in breadcrumb or
                any(kw.lower() in breadcrumb or kw.lower() in content for kw in domain_kws)
            )

            if is_structural_match:
                boosted_score = round(base_score + port_boost, 4)
                item["rerank_score"] = boosted_score
                item["adaptive_boost"] = port_boost
            else:
                item["adaptive_boost"] = 0.0

            boosted_chunks.append(item)

        # Re-sort chunks based on boosted scores
        boosted_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return boosted_chunks
