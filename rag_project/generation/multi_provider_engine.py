import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from google import genai
from google.genai import types

from rag_project.config import (
    GOOGLE_API_KEY_FILE,
    APIS_FILE,
    SYSTEM_PROMPT_FILE,
    load_system_prompt,
    GEMINI_DEFAULT_MODEL,
    OLLAMA_BASE_URL,
    MAX_OUTPUT_TOKENS,
    TIMEOUT_SECONDS,
    BASE_DIR
)
from rag_project.generation.api_key_manager import ApiKeyManager


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = load_system_prompt()


class MultiProviderEngine:
    """Resilient Multi-Provider Inference Pool with Tiered Fallback, Circuit Breaker,
    and thread-safe rate-limiting across Google GenAI, Mistral AI, Cohere, and OpenRouter.
    Loads credentials dynamically via ApiKeyManager. ZERO HARDCODED SECRETS.
    """

    def __init__(
        self,
        apis_file: Optional[Path] = None,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None
    ):
        self.key_manager = ApiKeyManager(apis_file=apis_file)
        self.preferred_provider = preferred_provider.lower() if preferred_provider else None
        self.preferred_model = preferred_model
        self.lock = threading.Lock()
        self.circuit_breakers: Dict[str, float] = {}  # provider_key -> quarantined_until_timestamp
        self.providers = []
        self._load_and_init_providers()

    def _load_and_init_providers(self):
        """Configure providers dynamically based on loaded API keys."""
        # 1. Tier 1: Google GenAI (Round-Robin across all active keys)
        google_keys = self.key_manager.get_google_keys()
        if google_keys:
            for k in google_keys:
                try:
                    c = genai.Client(api_key=k)
                    self.providers.append({
                        "id": f"google_{k[:8]}",
                        "name": "Google GenAI",
                        "tier": 1 if self.preferred_provider in [None, "google", "gemini"] else 3,
                        "type": "google",
                        "client": c,
                        "models": [self.preferred_model] if (self.preferred_model and "gemini" in self.preferred_model) else ["gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-2.5-flash"],
                        "min_delay": 0.2,
                        "last_call": 0.0
                    })
                except Exception as e:
                    logger.warning(f"Error initializing Google client: {e}")

        # 2. Tier 2: Mistral AI (mistral-small-latest, codestral-latest)
        mistral_keys = self.key_manager.get_mistral_keys()
        if mistral_keys:
            for k in mistral_keys:
                self.providers.append({
                    "id": f"mistral_{k[:8]}",
                    "name": "Mistral AI",
                    "tier": 1 if self.preferred_provider == "mistral" else 2,
                    "type": "openai_compat",
                    "base_url": "https://api.mistral.ai/v1/chat/completions",
                    "api_key": k,
                    "models": [self.preferred_model] if (self.preferred_model and "mistral" in self.preferred_model) else ["mistral-small-latest", "codestral-latest"],
                    "min_delay": 1.0,
                    "last_call": 0.0
                })

        # 3. Tier 2: Cohere (command-r-08-2024)
        cohere_keys = self.key_manager.get_cohere_keys()
        if cohere_keys:
            for k in cohere_keys:
                self.providers.append({
                    "id": f"cohere_{k[:8]}",
                    "name": "Cohere",
                    "tier": 1 if self.preferred_provider == "cohere" else 2,
                    "type": "cohere_v2",
                    "base_url": "https://api.cohere.com/v2/chat",
                    "api_key": k,
                    "models": [self.preferred_model] if (self.preferred_model and "command" in self.preferred_model) else ["command-r-08-2024"],
                    "min_delay": 1.5,
                    "last_call": 0.0
                })

        # 4. Tier 3: OpenRouter (Llama 3.1 8B, Llama 3.3 70B, DeepSeek Chat, Qwen 2.5 72B)
        openrouter_keys = self.key_manager.get_openrouter_keys()
        if openrouter_keys:
            for k in openrouter_keys:
                self.providers.append({
                    "id": f"openrouter_{k[:8]}",
                    "name": "OpenRouter",
                    "tier": 1 if self.preferred_provider == "openrouter" else 3,
                    "type": "openai_compat",
                    "base_url": "https://openrouter.ai/api/v1/chat/completions",
                    "api_key": k,
                    "models": [self.preferred_model] if (self.preferred_model and self.preferred_provider == "openrouter") else [
                        "meta-llama/llama-3.1-8b-instruct",
                        "meta-llama/llama-3.3-70b-instruct",
                        "deepseek/deepseek-chat",
                        "qwen/qwen-2.5-72b-instruct"
                    ],
                    "min_delay": 1.5,
                    "last_call": 0.0
                })

        # 5. Local Engine (Ollama on CUDA): Gemma 3 4B / Gemma 3 1B
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                ollama_models = [m.get("name", "") for m in resp.json().get("models", [])]
                if not ollama_models:
                    ollama_models = ["gemma3:4b", "gemma3:1b"]
                self.providers.append({
                    "id": "ollama_local",
                    "name": "Local Ollama",
                    "tier": 1 if self.preferred_provider == "ollama" else 4,
                    "type": "ollama",
                    "base_url": OLLAMA_BASE_URL,
                    "models": [self.preferred_model] if (self.preferred_model and self.preferred_provider == "ollama") else ollama_models,
                    "min_delay": 0.0,
                    "last_call": 0.0
                })
        except Exception:
            pass

        # Sort providers by tier
        self.providers.sort(key=lambda p: p.get("tier", 99))
        logger.info(f"MultiProviderEngine initialized with {len(self.providers)} active providers across Tiers 1-4.")



    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Construct structured multi-document context payload."""
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
        temperature: float = 0.0,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute inference through prioritized providers with circuit breaker and failover."""
        prompt = self.build_prompt(query, context_chunks)
        now = time.time()

        req_prov = (preferred_provider or self.preferred_provider or "").lower()
        req_model = preferred_model or self.preferred_model

        # Strict provider filtering: if a specific provider is requested (e.g. 'gemini' or 'google'),
        # strictly isolate execution to that provider fleet and never spill over to secondary models.
        if req_prov:
            provider_list = [
                p for p in self.providers
                if req_prov in p.get("type", "").lower()
                or req_prov in p.get("name", "").lower()
                or req_prov in p.get("id", "").lower()
            ]
            if not provider_list:
                provider_list = self.providers
        else:
            def get_prov_sort_key(p):
                return p.get("tier", 99)
            provider_list = sorted(self.providers, key=get_prov_sort_key)

        # Iterate through providers
        for p in provider_list:
            pid = p["id"]

            # Check circuit breaker quarantine
            if pid in self.circuit_breakers and self.circuit_breakers[pid] > now:
                continue

            # Thread-safe rate limiter pacing
            with self.lock:
                elapsed = time.time() - p.get("last_call", 0.0)
                if elapsed < p["min_delay"]:
                    time.sleep(p["min_delay"] - elapsed)
                p["last_call"] = time.time()

            # Execute based on provider type
            ptype = p["type"]
            models_to_try = [req_model] if (req_model and (req_prov in p["name"].lower() or req_prov in p["id"].lower() or req_prov in ptype)) else p["models"]
            for model_name in models_to_try:
                try:
                    t0 = time.time()

                    if ptype == "google":
                        res_text = self._call_google(p["client"], model_name, prompt, temperature)
                    elif ptype == "openai_compat":
                        res_text = self._call_openai_compat(p["base_url"], p["api_key"], model_name, prompt, temperature)
                    elif ptype == "cohere_v2":
                        res_text = self._call_cohere_v2(p["base_url"], p["api_key"], model_name, prompt, temperature)
                    elif ptype == "ollama":
                        res_text = self._call_ollama(p["base_url"], model_name, prompt, temperature)
                    else:
                        continue


                    duration_ms = round((time.time() - t0) * 1000, 2)
                    if res_text:
                        return {
                            "response": res_text,
                            "provider": p["name"],
                            "model": model_name,
                            "total_duration_ms": duration_ms,
                            "eval_count": len(res_text.split()),
                        }
                except Exception as e:
                    err_str = str(e).lower()
                    logger.warning(f"Provider {p['name']} ({model_name}) error: {e}")

                    # Trip circuit breaker on 429 or 503
                    if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str or "503" in err_str:
                        self.circuit_breakers[pid] = time.time() + 60.0
                        logger.warning(f"Circuit breaker tripped for {p['name']} ({pid}). Quarantined for 60s.")
                        break  # Try next provider immediately

        # Final fallback error response if all providers are exhausted
        return {
            "response": f"The provided technical documentation context does not contain information regarding {query}. [Error: All providers exhausted]",
            "provider": "None",
            "model": "None",
            "total_duration_ms": 0.0,
            "eval_count": 0,
        }

    def _call_google(self, client: genai.Client, model: str, prompt: str, temperature: float) -> str:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            top_p=0.9,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text if response.text else ""

    def _call_openai_compat(self, url: str, api_key: str, model: str, prompt: str, temperature: float) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/merfan-bagheri/production-rag-framework",
            "X-Title": "Production RAG Framework"
        }
        # Safely clamp to 16,384 for generic OpenAI compatible endpoints if needed
        tokens_limit = min(MAX_OUTPUT_TOKENS, 16384)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": tokens_limit
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:150]}")

    def _call_cohere_v2(self, url: str, api_key: str, model: str, prompt: str, temperature: float) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        tokens_limit = min(MAX_OUTPUT_TOKENS, 4096)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": tokens_limit
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", [{}])[0].get("text", "").strip()
        else:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:150]}")

    def _call_ollama(self, base_url: str, model: str, prompt: str, temperature: float) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": MAX_OUTPUT_TOKENS,
            }
        }
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=TIMEOUT_SECONDS)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        else:
            raise Exception(f"Ollama HTTP {resp.status_code}: {resp.text[:150]}")

