import logging
import re
import time
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from rag_project.config import (
    GOOGLE_API_KEY_FILE,
    GEMINI_DEFAULT_MODEL,
    GEMINI_FALLBACK_MODEL,
    QUERY_REFORMULATION_CONFIG
)
from rag_project.generation.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REFORMULATION_SYSTEM_PROMPT = QUERY_REFORMULATION_CONFIG.get(
    "system_prompt",
    """You are a High-Precision Query Condensation Engine for a Multi-Document Technical Knowledge Base.

YOUR TASK:
Given the Multi-Turn Conversation History and the latest Follow-up User Query:
1. Resolve all anaphoric references, pronouns (it, its, this, that, they, these), and ordinal references (step 2, option 4, table 3, the second one).
2. Retain the specific document, system, or component context discussed in previous turns if the user asks a follow-up about that topic.
3. Formulate a single, self-contained, unambiguous standalone search query optimized for dense vector and lexical BM25 retrieval.
4. DO NOT answer the question. Output ONLY the condensed standalone query string and nothing else."""
)

ANAPHORIC_PATTERNS = QUERY_REFORMULATION_CONFIG.get(
    "anaphoric_patterns",
    [
        r"\b(it|its|they|them|their|this|that|these|those)\b",
        r"\b(what about|how about|tell me more|explain more)\b",
        r"\b(step|option|item|table|figure|section|part)\s+\d+\b",
        r"\b(the\s+(?:first|second|third|fourth|fifth|last|previous|former|latter))\b",
        r"\b(where\s+must\s+(?:that|the)\s+file\s+reside)\b",
        r"\b(and\s+(?:what|how|why|where))\b",
        r"^(why|how|what|where)\s+(?:is|are|do|does|did|was|were)\s+(?:it|that|this)\b",
    ]
)

class QueryReformulator:
    """Zero-overhead history-aware query condensation engine for multi-turn RAG dialogues."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        model_name: str = GEMINI_DEFAULT_MODEL,
        fallback_model: str = GEMINI_FALLBACK_MODEL,
    ):
        self.gemini_client = gemini_client or GeminiClient(model_name=model_name, fallback_model=fallback_model)
        self.clients = self.gemini_client.clients
        self.model_name = model_name
        self.fallback_model = fallback_model

    def has_anaphoric_reference(self, query: str) -> bool:
        """Fast heuristic check for coreference markers requiring reformulation."""
        q_lower = query.lower()
        return any(re.search(p, q_lower) for p in ANAPHORIC_PATTERNS)

    def reformulate(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Reformulate conversational follow-up into a self-contained search query."""
        if not history:
            return {
                "reformulated_query": query,
                "was_reformulated": False,
                "latency_ms": 0.0,
                "method": "bypass_empty_history"
            }

        # Fast heuristic bypass if standalone
        if not self.has_anaphoric_reference(query) and len(query.split()) > 7:
            # Check if query already mentions specific document identifiers
            if any(doc in query.upper() for doc in ["PG036", "PG058", "PG065", "UG380", "UG389", "UG682", "BLOCK MEMORY", "DISTRIBUTED MEMORY"]):
                return {
                    "reformulated_query": query,
                    "was_reformulated": False,
                    "latency_ms": 0.0,
                    "method": "heuristic_bypass_standalone"
                }

        # Format chat history (last 3 turns max for fast execution)
        recent_history = history[-6:]
        history_lines = []
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            if len(content) > 350:
                content = content[:350] + "..."
            history_lines.append(f"{role}: {content}")

        history_text = "\n".join(history_lines)
        prompt = f"""Conversation History:
---------------------
{history_text}
---------------------

Latest Follow-up User Query:
{query}

Standalone Search Query:"""

        t0 = time.time()
        for client in self.clients:
            for model in [self.model_name, self.fallback_model]:
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=REFORMULATION_SYSTEM_PROMPT,
                        temperature=0.0,
                        top_p=0.5,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    )
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    latency = round((time.time() - t0) * 1000, 2)
                    text = response.text.strip().strip('"').strip("'") if response.text else query
                    # Clean any prefixes
                    if text.lower().startswith("standalone search query:"):
                        text = text[len("standalone search query:"):].strip()
                    logger.info(f"Query reformulated ({latency}ms): '{query}' -> '{text}'")
                    return {
                        "reformulated_query": text,
                        "was_reformulated": True,
                        "latency_ms": latency,
                        "method": f"gemini_{model}"
                    }
                except Exception as e:
                    logger.warning(f"Query reformulation error ({model}): {e}")
                    continue

        return {
            "reformulated_query": query,
            "was_reformulated": False,
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "method": "fallback_raw"
        }
