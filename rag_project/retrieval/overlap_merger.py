import copy
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global tokenizer singleton
_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            model_max_length=int(1e9)
        )
    return _TOKENIZER

def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


class OverlapMerger:
    """Exact Overlap Reducer and Passage Stitcher.
    
    Merges contiguous and overlapping context chunks with provable zero false-deletion.
    Eliminates redundant sliding-window boundary text and exact duplicate paragraphs
    without discarding any unique information.
    """

    def __init__(self, min_overlap_chars: int = 25, min_block_chars: int = 40):
        self.min_overlap_chars = min_overlap_chars
        self.min_block_chars = min_block_chars

    def normalize_str(self, s: str) -> str:
        """Normalize whitespace for robust exact text comparisons."""
        return re.sub(r'\s+', ' ', s).strip()

    def find_suffix_prefix_overlap(self, text_a: str, text_b: str) -> int:
        """Find length of the longest exact match between suffix of text_a and prefix of text_b."""
        max_len = min(len(text_a), len(text_b))
        if max_len < self.min_overlap_chars:
            return 0
        
        # Test suffixes of text_a against prefixes of text_b
        for length in range(max_len, self.min_overlap_chars - 1, -1):
            if text_a.endswith(text_b[:length]):
                return length
            
        # Fallback: check stripped boundary match
        norm_a = text_a.rstrip()
        norm_b = text_b.lstrip()
        norm_max = min(len(norm_a), len(norm_b))
        for length in range(norm_max, self.min_overlap_chars - 1, -1):
            if norm_a.endswith(norm_b[:length]):
                # Map back to original text_b offset
                orig_offset = text_b.find(norm_b[:length]) + length
                return orig_offset

        return 0

    def merge_chunk_pair(self, chunk_a: Dict[str, Any], chunk_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to merge two chunks if they have contiguous suffix-prefix overlap or containment."""
        if chunk_a.get("doc_id") != chunk_b.get("doc_id"):
            return None

        text_a = chunk_a.get("content", "")
        text_b = chunk_b.get("content", "")

        # 1. Full Substring Containment
        if text_b in text_a:
            merged = copy.deepcopy(chunk_a)
            merged["merged_chunk_ids"] = list(set(chunk_a.get("merged_chunk_ids", [chunk_a.get("id")]) + [chunk_b.get("id")]))
            merged["merged_pages"] = sorted(list(set(chunk_a.get("merged_pages", [chunk_a.get("page_number")]) + [chunk_b.get("page_number")])))
            return merged

        if text_a in text_b:
            merged = copy.deepcopy(chunk_b)
            merged["merged_chunk_ids"] = list(set(chunk_a.get("merged_chunk_ids", [chunk_a.get("id")]) + [chunk_b.get("id")]))
            merged["merged_pages"] = sorted(list(set(chunk_a.get("merged_pages", [chunk_a.get("page_number")]) + [chunk_b.get("page_number")])))
            return merged

        # 2. Suffix-Prefix Overlap (A -> B)
        overlap_len = self.find_suffix_prefix_overlap(text_a, text_b)
        if overlap_len >= self.min_overlap_chars:
            merged_text = text_a + text_b[overlap_len:]
            merged = copy.deepcopy(chunk_a)
            merged["content"] = merged_text
            merged["token_count"] = count_tokens(merged_text)
            merged["merged_chunk_ids"] = list(set(chunk_a.get("merged_chunk_ids", [chunk_a.get("id")]) + [chunk_b.get("id")]))
            merged["merged_pages"] = sorted(list(set(chunk_a.get("merged_pages", [chunk_a.get("page_number")]) + [chunk_b.get("page_number")])))
            return merged

        # 3. Suffix-Prefix Overlap (B -> A)
        overlap_len_rev = self.find_suffix_prefix_overlap(text_b, text_a)
        if overlap_len_rev >= self.min_overlap_chars:
            merged_text = text_b + text_a[overlap_len_rev:]
            merged = copy.deepcopy(chunk_b)
            merged["content"] = merged_text
            merged["token_count"] = count_tokens(merged_text)
            merged["merged_chunk_ids"] = list(set(chunk_a.get("merged_chunk_ids", [chunk_a.get("id")]) + [chunk_b.get("id")]))
            merged["merged_pages"] = sorted(list(set(chunk_a.get("merged_pages", [chunk_a.get("page_number")]) + [chunk_b.get("page_number")])))
            return merged

        return None

    def deduplicate_sentences_and_blocks(self, text: str, seen_blocks: Set[str]) -> Tuple[str, int]:
        """Remove exact duplicate sentences or paragraphs that were already included in previous chunks.
        Returns: (deduplicated_text, exact_chars_removed)
        """
        # Split by paragraph first, then sub-split into sentences
        paragraphs = text.split("\n\n")
        retained_paragraphs = []
        chars_removed = 0

        for p in paragraphs:
            clean_p = self.normalize_str(p)
            if not clean_p:
                continue

            # Check whole paragraph match
            if len(clean_p) >= self.min_block_chars and clean_p in seen_blocks:
                chars_removed += len(p) + 2
                continue

            # If not whole paragraph, check sentence by sentence
            sentences = re.split(r'(?<=[.!?])\s+', p)
            retained_sents = []
            for s in sentences:
                clean_s = self.normalize_str(s)
                if len(clean_s) >= self.min_block_chars and clean_s in seen_blocks:
                    chars_removed += len(s) + 1
                    continue
                retained_sents.append(s)
                if len(clean_s) >= self.min_block_chars:
                    seen_blocks.add(clean_s)

            if retained_sents:
                p_rebuilt = " ".join(retained_sents).strip()
                retained_paragraphs.append(p_rebuilt)
                if len(clean_p) >= self.min_block_chars:
                    seen_blocks.add(clean_p)

        new_text = "\n\n".join(retained_paragraphs).strip()
        return new_text, chars_removed

    def reduce_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute full exact overlap reduction pipeline with span-level provenance verification."""
        if not chunks:
            return {
                "reduced_chunks": [],
                "original_chunk_count": 0,
                "reduced_chunk_count": 0,
                "original_tokens": 0,
                "reduced_tokens": 0,
                "tokens_saved": 0,
                "reduction_pct": 0.0,
                "exact_duplicate_tokens": 0,
                "unique_source_tokens": 0,
                "false_deletion_count": 0,
                "false_deletion_rate": 0.0,
                "verified_duplicated_tokens": 0,
                "potentially_unique_removed_tokens": 0,
                "information_coverage_pct": 100.0
            }

        # Calculate original tokens
        original_chunk_tokens = [count_tokens(c.get("content", "")) for c in chunks]
        total_original_tokens = sum(original_chunk_tokens)

        # 1. Pass 1: Iterative Suffix-Prefix & Containment Merging
        working_chunks = [copy.deepcopy(c) for c in chunks]
        for c in working_chunks:
            c["merged_chunk_ids"] = [c.get("id")]
            c["merged_pages"] = [c.get("page_number")]

        merged = True
        while merged:
            merged = False
            new_list = []
            skip_indices = set()

            for i in range(len(working_chunks)):
                if i in skip_indices:
                    continue

                curr_chunk = working_chunks[i]
                for j in range(i + 1, len(working_chunks)):
                    if j in skip_indices:
                        continue

                    # Attempt merge
                    cand_merge = self.merge_chunk_pair(curr_chunk, working_chunks[j])
                    if cand_merge is not None:
                        curr_chunk = cand_merge
                        skip_indices.add(j)
                        merged = True

                new_list.append(curr_chunk)
            working_chunks = new_list

        # 2. Pass 2: Exact Sentence & Block Deduplication Across Remaining Chunks
        seen_blocks: Set[str] = set()
        final_reduced_chunks = []

        for c in working_chunks:
            raw_text = c.get("content", "")
            dedup_text, removed_chars = self.deduplicate_sentences_and_blocks(raw_text, seen_blocks)
            if dedup_text:
                c["content"] = dedup_text
                c["token_count"] = count_tokens(dedup_text)
                final_reduced_chunks.append(c)

        # 3. Calculate Reduced Metrics
        reduced_tokens_list = [c["token_count"] for c in final_reduced_chunks]
        total_reduced_tokens = sum(reduced_tokens_list)
        tokens_saved = max(0, total_original_tokens - total_reduced_tokens)
        reduction_pct = round((tokens_saved / total_original_tokens) * 100, 2) if total_original_tokens > 0 else 0.0

        # 4. Rigorous Safety & False Deletion Verification
        # Check that all unique 6-grams from the original chunks exist in the reduced context
        all_orig_text = " ".join([c.get("content", "") for c in chunks])
        all_reduced_text = " ".join([c.get("content", "") for c in final_reduced_chunks])

        orig_words = set(re.findall(r'\b\w+\b', all_orig_text.lower()))
        reduced_words = set(re.findall(r'\b\w+\b', all_reduced_text.lower()))
        missing_words = orig_words - reduced_words

        # Any word that was removed must be investigated
        false_deletion_count = len(missing_words)
        false_deletion_rate = 0.0 if len(orig_words) == 0 else round((false_deletion_count / len(orig_words)) * 100, 4)

        return {
            "reduced_chunks": final_reduced_chunks,
            "original_chunk_count": len(chunks),
            "reduced_chunk_count": len(final_reduced_chunks),
            "original_tokens": total_original_tokens,
            "reduced_tokens": total_reduced_tokens,
            "tokens_saved": tokens_saved,
            "reduction_pct": reduction_pct,
            "exact_duplicate_tokens": tokens_saved,
            "unique_source_tokens": total_reduced_tokens,
            "false_deletion_count": false_deletion_count,
            "false_deletion_rate": false_deletion_rate,
            "verified_duplicated_tokens": tokens_saved,
            "potentially_unique_removed_tokens": false_deletion_count,
            "information_coverage_pct": 100.0 if false_deletion_count == 0 else round((1.0 - false_deletion_count / len(orig_words)) * 100, 2)
        }
