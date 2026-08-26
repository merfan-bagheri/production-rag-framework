import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from rag_project.config import (
    DEFAULT_PDF_PATH,
    GOOGLE_API_KEY_FILE,
    GEMINI_DEFAULT_MODEL,
    GEMINI_API_KEY
)
from rag_project.ingestion.pdf_parser import PDFStructureParser
from rag_project.ingestion.chunker import StructuralChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENRICHMENT_PROMPT = """You are a Principal FPGA & EDA Technical Documentation Engineer.
Your task is to transform the following raw technical hardware text extracted from a Xilinx LogiCORE IP Product Guide into pristine, structure-aware technical Markdown.

STRICT TRANSFORMATION RULES:
1. PRESERVE EVERY TECHNICAL DETAIL: Do NOT summarize, drop, or alter any pin names, widths, active levels, register options, timing numbers, or device family constraints.
2. PERFECT MARKDOWN TABLES: Reconstruct all port descriptions, register tables, timing parameters, and COE formats into clean, well-aligned Markdown tables with standard headers (e.g. | Signal | Direction | Width | Description & Latency |).
3. EXPLICIT SIGNAL TAGS: At the top of the chunk, output a metadata tag: `Signals: [signal1, signal2, ...]` listing every hardware pin/signal mentioned.
4. HIERARCHICAL BREADCRUMB: Retain the section breadcrumb header.
5. NO HALLUCINATION: Include ONLY details present in the raw input text.

Raw Section Input:
Section Breadcrumb: {breadcrumb}
Page Number: {page_number}
Content:
{content}

Output the structured, cleaned Markdown directly:"""

class GeminiDocumentEnricher:
    """Enriches raw PDF chunks using Gemini 3.7 Flash for ultra-high-fidelity hardware RAG retrieval."""

    def __init__(
        self,
        api_key_file: Path = GOOGLE_API_KEY_FILE,
        model_name: str = GEMINI_DEFAULT_MODEL,
    ):
        self.api_keys = self._load_keys(api_key_file)
        self.current_key_idx = 0
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_keys[0])

    def _load_keys(self, key_file: Path) -> List[str]:
        keys = []
        if GEMINI_API_KEY:
            keys.append(GEMINI_API_KEY.strip())
        if key_file.exists():
            with open(key_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        keys.append(line)
        return keys if keys else [""]

    def _rotate_key(self):
        if len(self.api_keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            logger.info(f"Rotating to API key #{self.current_key_idx + 1}")
            self.client = genai.Client(api_key=self.api_keys[self.current_key_idx])

    def enrich_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich an individual chunk with Gemini-structured tables and signal tags."""
        content = chunk.get("content", "")
        # If chunk is very small or plain prose, keep as is
        if len(content.strip()) < 80:
            return chunk

        breadcrumb = chunk.get("breadcrumb", "General")
        page_num = chunk.get("page_number", 1)

        prompt = ENRICHMENT_PROMPT.format(
            breadcrumb=breadcrumb,
            page_number=page_num,
            content=content
        )

        for attempt in range(len(self.api_keys) * 2):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        top_p=0.95,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    )
                )
                if response.text and len(response.text.strip()) > 50:
                    enriched_content = response.text.strip()
                    
                    # Extract signals tag if generated
                    signals_match = re.search(r"Signals:\s*\[(.*?)\]", enriched_content)
                    signals_list = []
                    if signals_match:
                        signals_list = [s.strip() for s in signals_match.group(1).split(",") if s.strip()]

                    enriched_chunk = dict(chunk)
                    enriched_chunk["content"] = enriched_content
                    meta = dict(chunk.get("metadata", {}))
                    meta["gemini_enriched"] = True
                    meta["signals"] = signals_list
                    enriched_chunk["metadata"] = meta
                    return enriched_chunk
            except Exception as e:
                logger.warning(f"Gemini chunk enrichment retry on key #{self.current_key_idx + 1}: {e}")
                self._rotate_key()
                time.sleep(1)

        # Fallback to original chunk if all retries fail
        return chunk

    def enrich_all_chunks(self, chunks: List[Dict[str, Any]], max_enriched: Optional[int] = None) -> List[Dict[str, Any]]:
        """Enrich critical hardware chunks (tables, pinouts, timing, and COE formats)."""
        logger.info(f"Starting Gemini enrichment on {len(chunks)} chunks...")
        enriched = []
        target_chunks = chunks[:max_enriched] if max_enriched else chunks

        for idx, c in enumerate(target_chunks, 1):
            c_type = c.get("content_type", "")
            content = c.get("content", "")
            
            # Prioritize chunks containing tables, signal names, registers, COE, or timing
            is_priority = (
                c_type == "table" or 
                any(sig in content.lower() for sig in ["spo", "qspo", "dpo", "qdpo", "coe", "latency", "register", "dpra", "spra", "srl16", "radix"])
            )
            
            if is_priority:
                logger.info(f"Enriching priority chunk {idx}/{len(target_chunks)}: [Page {c.get('page_number')}] {c.get('breadcrumb')}")
                en_chunk = self.enrich_chunk(c)
                enriched.append(en_chunk)
            else:
                enriched.append(c)

        logger.info(f"Gemini enrichment completed for {len(enriched)} chunks.")
        return enriched
