import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from rag_project.generation.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ChatSession:
    """Stateful Multi-Turn Conversational Session Manager with Multi-Document Retention."""

    def __init__(
        self,
        pipeline: Optional[RAGPipeline] = None,
        max_history_turns: int = 4,
        auto_k_default: bool = True
    ):
        self.pipeline = pipeline or RAGPipeline()
        self.max_history_turns = max_history_turns
        self.auto_k_default = auto_k_default
        self.history: List[Dict[str, Any]] = []
        self.session_id = f"session_{int(time.time())}"

    def ask(
        self,
        query: str,
        top_k: Optional[Union[int, str]] = None,
        auto_k: Optional[bool] = None,
        temperature: float = 0.1,
        doc_filter: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Execute a conversational turn with history-aware query condensation and adaptive retrieval."""
        effective_auto_k = self.auto_k_default if auto_k is None else auto_k
        if top_k == "auto":
            effective_auto_k = True
            effective_top_k = 7
        elif isinstance(top_k, int):
            effective_top_k = top_k
            effective_auto_k = False if auto_k is None else auto_k
        else:
            effective_top_k = 7

        # Format history turns for reformulation
        history_tuples = []
        for turn in self.history[-self.max_history_turns:]:
            history_tuples.append({"role": "user", "content": turn["query"]})
            history_tuples.append({"role": "assistant", "content": turn["answer"]})

        # Process query through pipeline with history
        result = self.pipeline.query(
            query=query,
            history=history_tuples,
            top_k_rerank=effective_top_k,
            auto_k=effective_auto_k,
            doc_filter=doc_filter,
            temperature=temperature
        )

        # Store turn in history
        turn_record = {
            "turn_index": len(self.history) + 1,
            "query": query,
            "reformulated_query": result.get("reformulated_query"),
            "route_info": result.get("route_info", {}),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "timings_ms": result.get("timings_ms", {}),
            "adaptive_strategy": result.get("adaptive_strategy", {}),
            "timestamp": time.time()
        }
        self.history.append(turn_record)

        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieve full conversation history."""
        return list(self.history)

    def reset(self):
        """Clear all conversation history."""
        self.history = []
        logger.info("Chat session history cleared.")

    def export_markdown(self) -> str:
        """Export session dialogue as formatted Markdown."""
        lines = [
            f"# Multi-Document Xilinx Knowledge Base Conversation Export",
            f"**Session ID:** {self.session_id}",
            f"**Total Turns:** {len(self.history)}",
            "\n---\n"
        ]

        for turn in self.history:
            lines.append(f"### 👤 User (Turn {turn['turn_index']}):")
            lines.append(turn['query'])
            if turn.get('reformulated_query'):
                lines.append(f"\n> *[History-Aware Rewritten Query: {turn['reformulated_query']}]*")
            lines.append(f"\n### 🤖 Assistant:")
            lines.append(turn['answer'])
            if turn.get('sources'):
                lines.append(f"\n**Verified Sources ({len(turn['sources'])} chunks):**")
                for s in turn['sources']:
                    doc_id = s.get('doc_id', 'PG036')
                    lines.append(f"- [{doc_id}] Page {s.get('page_number', '?')} • {s.get('breadcrumb', 'N/A')} (Rerank: {s.get('rerank_score', 'N/A')})")
            lines.append("\n---\n")

        return "\n".join(lines)
