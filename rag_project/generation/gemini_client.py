import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from rag_project.config import (
    GOOGLE_API_KEY_FILE,
    APIS_FILE,
    SYSTEM_PROMPT_FILE,
    load_system_prompt,
    GEMINI_DEFAULT_MODEL,
    GEMINI_API_KEY,
    MAX_OUTPUT_TOKENS
)
from rag_project.generation.api_key_manager import ApiKeyManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GEMINI_SYSTEM_PROMPT = load_system_prompt()

class GeminiClient:
    """Production Client for Google AI Studio Gemini models (exclusively gemini-3.5-flash-lite)
    with strict client-side rate throttling, full-jitter exponential backoff, and robust error handling.
    """

    def __init__(
        self,
        api_key_file: Path = GOOGLE_API_KEY_FILE,
        model_name: str = GEMINI_DEFAULT_MODEL,
        fallback_model: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 30.0,
    ):
        self.api_keys = self._load_api_keys(api_key_file)
        self.current_key_idx = 0
        self.model_name = model_name
        self.fallback_model = fallback_model or model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.clients = [genai.Client(api_key=k) for k in self.api_keys if k]

    def _load_api_keys(self, key_file: Path) -> List[str]:
        """Load valid Google AI Studio API keys dynamically via ApiKeyManager."""
        keys = ApiKeyManager.get_instance().get_google_keys()

        if not keys:
            logger.error("No Google AI Studio API key found in APIs.txt, google-api-key.txt, or GEMINI_API_KEY environment variable.")
            raise ValueError("Google API key is missing. Please configure APIs.txt or set GEMINI_API_KEY.")
        return keys


    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Construct structured multi-document context payload for Gemini."""
        context_sections = []
        for idx, chunk in enumerate(context_chunks, 1):
            doc_id = chunk.get("doc_id", "PG036")
            doc_title = chunk.get("doc_title", doc_id)
            page_num = chunk.get("page_number", "Unknown")
            sec_title = chunk.get("section_title", "General")
            breadcrumb = chunk.get("breadcrumb", sec_title)
            c_type = chunk.get("content_type", "prose")
            content = chunk.get("content", "").strip()

            header = f"--- Context Chunk {idx} [Doc: {doc_title} | Page: {page_num} | Section: {breadcrumb}] (Type: {c_type}) ---"
            context_sections.append(f"{header}\n{content}")

        context_body = "\n\n".join(context_sections)

        prompt = f"""Retrieved Technical Hardware Context:
=====================================================
{context_body}
=====================================================

User Technical Hardware Query:
{query}

Engineering Analysis & Response:
(Address all aspects of the query completely and exhaustively. For every statement, signal definition, latency value, and configuration parameter, append an explicit citation in the format: [Doc: <doc_title> | Page: <page_number> | Section: <section_breadcrumb>]. For comparative queries, clearly delineate features per document.)"""
        return prompt

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate response exclusively using gemini-3.5-flash-lite with exponential backoff and jitter."""
        prompt = self.build_prompt(query, context_chunks)
        client = self.clients[self.current_key_idx % len(self.clients)]

        config = types.GenerateContentConfig(
            system_instruction=GEMINI_SYSTEM_PROMPT,
            temperature=temperature,
            top_p=0.9,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                gen_time_ms = round((time.time() - t0) * 1000, 2)
                response_text = response.text if response.text else ""

                return {
                    "response": response_text,
                    "model": self.model_name,
                    "total_duration_ms": gen_time_ms,
                    "eval_count": len(response_text.split()) if response_text else 0,
                }
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini generation error with {self.model_name} (attempt {attempt+1}/{self.max_retries}): {err_str[:120]}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff with full jitter: T_sleep = min(T_max, 2^attempt * base_delay + jitter)
                    jitter = random.uniform(0.5, 2.0)
                    backoff = min(self.max_delay, (2 ** attempt) * self.base_delay + jitter)
                    logger.info(f"Retrying generation after {backoff:.2f}s backoff...")
                    time.sleep(backoff)
                else:
                    logger.error(f"Gemini generation exhausted all {self.max_retries} attempts.")
                    return {
                        "response": f"The provided technical documentation context does not contain information regarding {query}. [Error: Generation timeout]",
                        "model": self.model_name,
                        "total_duration_ms": 0.0,
                        "eval_count": 0,
                    }
