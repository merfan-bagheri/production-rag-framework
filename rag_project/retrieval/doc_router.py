import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from rag_project.config import (
    DOCUMENT_REGISTRY,
    COMPARISON_PATTERNS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_digits(text: str) -> str:
    """Normalize Persian and Arabic digits to standard ASCII numerals."""
    fa_digits = "۰۱۲۳۴۵۶۷۸۹"
    ar_digits = "٠١٢٣٤٥٦٧٨٩"
    for i in range(10):
        text = text.replace(fa_digits[i], str(i)).replace(ar_digits[i], str(i))
    return text

class PageIntentParser:
    """Deterministic parser for extracting explicit page numbers, document bindings, and section targets."""

    @staticmethod
    def extract_page_targets(query: str, default_docs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract explicit page numbers with document context from query.
        Handles English, Persian, single pages, page ranges, and multi-doc explicit references.
        """
        norm_query = normalize_digits(query)
        q_lower = norm_query.lower()

        # 1. Identify all mentioned docs
        detected_docs = []
        for doc_id, info in DOCUMENT_REGISTRY.items():
            for pat in info["patterns"]:
                if re.search(pat, q_lower):
                    detected_docs.append(doc_id)
                    break
        detected_docs = list(set(detected_docs))
        if not detected_docs and default_docs:
            detected_docs = list(default_docs)

        # 2. Extract page patterns
        # Patterns:
        # - "page 30 in UG389", "page 30 of UG389", "UG389 page 30", "UG389 p. 30"
        # - "صفحه ۳۰ در UG389", "صفحه 30", "صفحات 30 تا 31"
        page_regex = r"\b(?:page|pages|p\.|صفحه|صفحات|صفحه‌ی)\s*(\d+)(?:\s*(?:to|-|تا|و)\s*(\d+))?\b"
        matches = re.finditer(page_regex, norm_query, re.IGNORECASE)

        targets = []
        for m in matches:
            start_p = int(m.group(1))
            end_p = int(m.group(2)) if m.group(2) else start_p

            if not (1 <= start_p <= 600):
                continue

            pages = list(range(start_p, min(end_p + 1, start_p + 3)))
            # Add adjacent neighbor (+1) for single page queries to ensure split tables/sections are captured
            if len(pages) == 1 and start_p < 550:
                pages.append(start_p + 1)

            # Determine associated document
            # Check if a specific doc appears right before or after this match
            surrounding_start = max(0, m.start() - 30)
            surrounding_end = min(len(norm_query), m.end() + 30)
            surrounding_text = norm_query[surrounding_start:surrounding_end].lower()

            bound_doc = None
            for d in detected_docs:
                if re.search(rf"\b{d.lower()}\b", surrounding_text):
                    bound_doc = d
                    break

            if not bound_doc and len(detected_docs) == 1:
                bound_doc = detected_docs[0]

            targets.append({
                "doc_id": bound_doc,
                "pages": pages,
                "requested_page": start_p,
                "raw_match": m.group(0)
            })

        return targets

class MultiDocRouter:
    """Pre-retrieval query classifier for document-focused filtering, page routing, and cross-document synthesis."""

    @staticmethod
    def route_query(query: str) -> Dict[str, Any]:
        """Analyze query to determine targeted document scope, explicit page targets, and multi-intent status."""
        norm_q = normalize_digits(query)
        q_lower = norm_q.lower()

        # Check for comparative intent
        is_comparative = any(re.search(pat, q_lower) for pat in COMPARISON_PATTERNS)

        # Detect matched documents
        matched_docs = []
        for doc_id, info in DOCUMENT_REGISTRY.items():
            for pat in info["patterns"]:
                if re.search(pat, q_lower):
                    matched_docs.append(doc_id)
                    break

        matched_docs = list(set(matched_docs))

        # Extract explicit page targets
        page_targets = PageIntentParser.extract_page_targets(norm_q, default_docs=matched_docs)
        is_page_inquiry = len(page_targets) > 0

        # Multi-intent check: query has explicit page target AND comparative intent OR multiple docs
        is_multi_intent = is_page_inquiry and (is_comparative or len(matched_docs) > 1 or len(norm_q.split()) > 10)

        # Classification Logic:
        if len(matched_docs) > 1:
            return {
                "scope": "cross_doc",
                "target_docs": matched_docs,
                "is_comparative": True,
                "is_page_inquiry": is_page_inquiry,
                "is_multi_intent": is_multi_intent,
                "page_targets": page_targets,
                "rationale": f"Cross-document comparative query detected. Targeting: {matched_docs}"
            }
        elif len(matched_docs) == 1:
            target_doc = matched_docs[0]
            return {
                "scope": "single_doc",
                "target_docs": [target_doc],
                "is_comparative": is_comparative,
                "is_page_inquiry": is_page_inquiry,
                "is_multi_intent": is_multi_intent,
                "page_targets": page_targets,
                "rationale": f"Single document scope targeted: {target_doc} ({DOCUMENT_REGISTRY[target_doc]['title']})"
            }
        else:
            return {
                "scope": "cross_doc",
                "target_docs": None,
                "is_comparative": is_comparative,
                "is_page_inquiry": is_page_inquiry,
                "is_multi_intent": is_multi_intent,
                "page_targets": page_targets,
                "rationale": "Global multi-document search across complete documentation repository."
            }
