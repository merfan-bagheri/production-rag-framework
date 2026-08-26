import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ApiKeyManager:
    """Unified, secure API Key Manager for Xilinx RAG Multi-Provider Fleet.
    Loads API keys dynamically from local configuration files (APIs.txt / google-api-key.txt)
    or environment variables. ZERO HARDCODED KEYS in codebase.
    """

    _INSTANCE: Optional['ApiKeyManager'] = None

    def __init__(self, apis_file: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parents[2]
        self.apis_file = apis_file or (self.project_root / "APIs.txt")
        self.google_key_file = self.project_root / "google-api-key.txt"
        self._credentials: Dict[str, List[str]] = {
            "google": [],
            "mistral": [],
            "cohere": [],
            "openrouter": [],
            "cerebras": []
        }
        self.reload()

    @classmethod
    def get_instance(cls) -> 'ApiKeyManager':
        if cls._INSTANCE is None:
            cls._INSTANCE = ApiKeyManager()
        return cls._INSTANCE

    def reload(self):
        """Scan environment variables and local files for API keys."""
        self._credentials = {
            "google": [],
            "mistral": [],
            "cohere": [],
            "openrouter": [],
            "cerebras": []
        }

        # 1. Load from Environment Variables
        if os.getenv("GEMINI_API_KEY"):
            self._credentials["google"].append(os.getenv("GEMINI_API_KEY").strip())
        if os.getenv("GOOGLE_API_KEY"):
            k = os.getenv("GOOGLE_API_KEY").strip()
            if k not in self._credentials["google"]:
                self._credentials["google"].append(k)
        if os.getenv("MISTRAL_API_KEY"):
            self._credentials["mistral"].append(os.getenv("MISTRAL_API_KEY").strip())
        if os.getenv("COHERE_API_KEY"):
            self._credentials["cohere"].append(os.getenv("COHERE_API_KEY").strip())
        if os.getenv("OPENROUTER_API_KEY"):
            self._credentials["openrouter"].append(os.getenv("OPENROUTER_API_KEY").strip())
        if os.getenv("CEREBRAS_API_KEY"):
            self._credentials["cerebras"].append(os.getenv("CEREBRAS_API_KEY").strip())

        # 2. Load from google-api-key.txt (Google AI Studio Key Pool)
        if self.google_key_file.exists():
            try:
                with open(self.google_key_file, "r", encoding="utf-8") as f:
                    for line in f:
                        k = line.strip()
                        if k and not k.startswith("#") and k not in self._credentials["google"]:
                            self._credentials["google"].append(k)
            except Exception as e:
                logger.warning(f"Could not read {self.google_key_file}: {e}")

        # 3. Load from APIs.txt (Unified Multi-Provider Config)
        if self.apis_file.exists():
            try:
                current_section = None
                with open(self.apis_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        
                        # Section headers
                        lower_line = line.lower()
                        if lower_line.startswith("google") or "gemini" in lower_line:
                            current_section = "google"
                        elif "mistral" in lower_line:
                            current_section = "mistral"
                        elif "cohere" in lower_line:
                            current_section = "cohere"
                        elif "openrouter" in lower_line:
                            current_section = "openrouter"
                        elif "cerebras" in lower_line:
                            current_section = "cerebras"
                        elif lower_line.startswith("http") or lower_line.endswith(":"):
                            # URL heading like https://console.mistral.ai/...
                            if "mistral" in lower_line:
                                current_section = "mistral"
                            elif "cohere" in lower_line:
                                current_section = "cohere"
                            elif "openrouter" in lower_line:
                                current_section = "openrouter"
                            elif "cerebras" in lower_line:
                                current_section = "cerebras"
                            elif "google" in lower_line or "generativelanguage" in lower_line:
                                current_section = "google"
                        elif current_section and len(line) > 8:
                            # Key value
                            if line not in self._credentials[current_section]:
                                self._credentials[current_section].append(line)
            except Exception as e:
                logger.warning(f"Could not read {self.apis_file}: {e}")

    def get_google_keys(self) -> List[str]:
        """Return list of valid Google GenAI API keys for rotational pooling."""
        return list(self._credentials.get("google", []))

    def get_mistral_keys(self) -> List[str]:
        return list(self._credentials.get("mistral", []))

    def get_cohere_keys(self) -> List[str]:
        return list(self._credentials.get("cohere", []))

    def get_openrouter_keys(self) -> List[str]:
        return list(self._credentials.get("openrouter", []))

    def get_first_key(self, provider: str) -> Optional[str]:
        keys = self._credentials.get(provider.lower(), [])
        return keys[0] if keys else None

    def get_status_summary(self) -> Dict[str, int]:
        return {p: len(keys) for p, keys in self._credentials.items()}
