import json
import logging
from typing import Any, Dict, List, Optional
import requests

from rag_project.config import (
    OLLAMA_BASE_URL,
    OLLAMA_PRIMARY_MODEL,
    OLLAMA_FALLBACK_MODEL,
    SYSTEM_PROMPT_FILE,
    load_system_prompt
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = load_system_prompt()

class OllamaClient:
    """Client for local Ollama LLM inference (Gemma 3 4B / 1B)."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        primary_model: str = OLLAMA_PRIMARY_MODEL,
        fallback_model: str = OLLAMA_FALLBACK_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.active_model = self.verify_and_select_model()

    def verify_and_select_model(self) -> str:
        """Verify Ollama availability and select best available Gemma 3 model."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                logger.info(f"Available Ollama models: {models}")
                for target in [self.primary_model, f"{self.primary_model}:latest"]:
                    if any(target in m for m in models):
                        logger.info(f"Selected primary model: {self.primary_model}")
                        return self.primary_model

                for target in [self.fallback_model, f"{self.fallback_model}:latest"]:
                    if any(target in m for m in models):
                        logger.warning(f"Primary model not found. Using fallback model: {self.fallback_model}")
                        return self.fallback_model

                if models:
                    logger.warning(f"Gemma 3 models not found. Using first available model: {models[0]}")
                    return models[0]
        except Exception as e:
            logger.error(f"Error connecting to Ollama at {self.base_url}: {e}")

        logger.warning(f"Defaulting to {self.primary_model}")
        return self.primary_model

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Construct a structured prompt containing ranked technical context chunks."""
        context_sections = []
        for idx, chunk in enumerate(context_chunks, 1):
            page_num = chunk.get("page_number", "Unknown")
            sec_title = chunk.get("section_title", "General")
            breadcrumb = chunk.get("breadcrumb", sec_title)
            c_type = chunk.get("content_type", "prose")
            content = chunk.get("content", "").strip()

            header = f"--- Context Chunk {idx} [Page {page_num}, Section: {breadcrumb}] (Type: {c_type}) ---"
            context_sections.append(f"{header}\n{content}")

        context_body = "\n\n".join(context_sections)

        prompt = f"""Retrieved Technical Hardware Context:
=====================================================
{context_body}
=====================================================

User Technical Hardware Query:
{query}

Engineering Analysis & Response:
(Address all aspects of the query. For every statement, signal definition, latency value, and configuration parameter, append an explicit citation in the format: [Page X, Section: Y]. If any requested feature or block is not in the context, explicitly state that it is not documented in PG036.)"""
        return prompt

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate response via Ollama /api/generate endpoint."""
        prompt = self.build_prompt(query, context_chunks)
        payload = {
            "model": self.active_model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_ctx": 8192,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")
            return {
                "response": response_text,
                "model": self.active_model,
                "total_duration_ms": round(data.get("total_duration", 0) / 1e6, 2),
                "eval_count": data.get("eval_count", 0),
            }
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return {
                "response": f"Error during generation: {e}",
                "model": self.active_model,
                "total_duration_ms": 0,
                "eval_count": 0,
            }
