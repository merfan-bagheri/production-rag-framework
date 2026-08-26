import logging
import re
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer

from rag_project.config import (
    MAX_CHUNK_CHARS,
    OVERLAP_CHARS,
    CHUNK_TARGET_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    MIN_CHUNK_TOKENS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fast subword tokenizer for accurate chunk sizing
_tokenizer = None

def get_chunker_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                model_max_length=int(1e9)
            )
        except Exception:
            _tokenizer = None
    return _tokenizer


class StructureAwareChunker:
    """Semantic Object-Aware Chunker for hardware manuals.
    Decomposes documents into atomic tables with attached footnotes, figure proxies with descriptive context,
    eliminates micro-chunks (< 50 tokens), and injects structured breadcrumb metadata.
    """

    def __init__(
        self,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        overlap_chars: int = OVERLAP_CHARS,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_tokens = min_chunk_tokens
        self.tok = get_chunker_tokenizer()

    def count_tokens(self, text: str) -> int:
        """Accurately count subword tokens using HuggingFace tokenizer."""
        if self.tok:
            try:
                return len(self.tok.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def is_table_block(self, text: str) -> bool:
        """Check if block is a markdown table."""
        return "|" in text and ("| ---" in text or "|:---" in text or text.count("\n|") >= 2)

    def extract_table_and_footnotes(self, text: str) -> Dict[str, Any]:
        """Extract table title, markdown table rows, and following footnotes."""
        lines = text.strip().split("\n")
        title = ""
        table_lines = []
        footnote_lines = []
        in_table = False
        after_table = False

        for line in lines:
            stripped = line.strip()
            if not in_table and not after_table:
                tbl_match = re.match(r"^(?:_|\*\*|#+\s*)?Table\s+(\d+[-–]\d+):?\s*(?:_|\*\*)?\s*(.*)$", stripped, re.IGNORECASE)
                if tbl_match:
                    num = tbl_match.group(1).replace("–", "-")
                    t_text = tbl_match.group(2).strip().strip("*_").strip()
                    title = f"Table {num}: {t_text}".strip(": ")
                    continue
                if stripped.startswith("|") and stripped.endswith("|"):
                    in_table = True
                    table_lines.append(line)
                    continue

            if in_table:
                if stripped.startswith("|") and stripped.endswith("|"):
                    table_lines.append(line)
                else:
                    in_table = False
                    after_table = True
                    if stripped:
                        footnote_lines.append(line)
                continue

            if after_table:
                if stripped:
                    footnote_lines.append(line)

        return {
            "title": title,
            "table_md": "\n".join(table_lines),
            "footnotes": "\n".join(footnote_lines),
            "is_valid": len(table_lines) >= 2
        }

    def split_table_safely(self, table_text: str, max_chars: int, title: str = "", footnotes: str = "") -> List[str]:
        """Split large markdown tables while retaining header row and separator in each sub-table, attaching title & notes."""
        lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
        if len(lines) <= 2:
            content = table_text
            if title:
                content = f"**{title}**\n\n{content}"
            if footnotes:
                content = f"{content}\n\n**Notes:**\n{footnotes}"
            return [content]

        header_lines = []
        data_lines = []
        for idx, line in enumerate(lines):
            if idx < 2:
                header_lines.append(line)
            else:
                data_lines.append(line)

        header = "\n".join(header_lines) + "\n"
        title_prefix = f"**{title}**\n\n" if title else ""
        chunks = []
        curr_chunk_lines = []
        curr_len = len(title_prefix) + len(header)

        for line in data_lines:
            line_len = len(line) + 1
            if curr_len + line_len > max_chars and curr_chunk_lines:
                chunk_body = title_prefix + header + "\n".join(curr_chunk_lines)
                chunks.append(chunk_body)
                curr_chunk_lines = [line]
                curr_len = len(title_prefix) + len(header) + line_len
            else:
                curr_chunk_lines.append(line)
                curr_len += line_len

        if curr_chunk_lines:
            last_body = title_prefix + header + "\n".join(curr_chunk_lines)
            if footnotes:
                last_body = f"{last_body}\n\n**Notes:**\n{footnotes}"
            chunks.append(last_body)

        return chunks if chunks else [table_text]

    def split_prose_safely(self, text: str, max_chars: int, overlap: int) -> List[str]:
        """Split long prose strictly along paragraph and sentence boundaries with full-sentence overlap."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p_str = p.strip()
            if not p_str:
                continue

            if len(current_chunk) + len(p_str) + 2 <= max_chars:
                current_chunk = f"{current_chunk}\n\n{p_str}".strip() if current_chunk else p_str
            else:
                if current_chunk:
                    # Check if current_chunk ends on incomplete clause (e.g. comma, dash, colon)
                    if re.search(r"[,;\-\(:\/]\s*$", current_chunk):
                        # Merge p_str directly into current_chunk rather than cutting mid-clause
                        current_chunk = f"{current_chunk}\n\n{p_str}"
                        if len(current_chunk) > max_chars * 1.5:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        continue

                    chunks.append(current_chunk)
                    # Sentence-aware overlap: find last complete sentence in current_chunk
                    sents = re.split(r"(?<=[.!?])\s+", current_chunk)
                    last_sent = sents[-1].strip() if sents else ""
                    if last_sent and len(last_sent) <= overlap:
                        current_chunk = f"{last_sent}\n\n{p_str}"
                    else:
                        current_chunk = p_str
                else:
                    # Single paragraph exceeds max_chars: split by complete sentences
                    sentences = re.split(r"(?<=[.!?])\s+", p_str)
                    for s in sentences:
                        s_clean = s.strip()
                        if not s_clean:
                            continue
                        if len(current_chunk) + len(s_clean) + 1 <= max_chars:
                            current_chunk = f"{current_chunk} {s_clean}".strip() if current_chunk else s_clean
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = s_clean

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def build_breadcrumb_header(
        self,
        doc_id: str,
        doc_title: str,
        breadcrumb: str,
        page_num: int,
        chunk_type: str,
        anchor: str = ""
    ) -> str:
        """Construct standard contextual breadcrumb header for all chunks."""
        anchor_part = f" | ANCHOR: {anchor}" if anchor else ""
        return f"[DOCUMENT: {doc_id} - {doc_title} | SECTION: {breadcrumb} | PAGE: {page_num} | TYPE: {chunk_type}{anchor_part}]"

    def chunk_section(
        self,
        section_data: Dict[str, Any],
        doc_id: str = "DOC",
        doc_title: str = "Manual"
    ) -> List[Dict[str, Any]]:
        """Chunk an extracted section into typed DOM primitives with atomic preservation and metadata headers."""
        raw_markdown = section_data.get("markdown", "").strip()
        if not raw_markdown:
            return []

        doc_name = section_data.get("document_name", "doc.pdf")
        page_num = section_data.get("page_number", 1)
        chapter = section_data.get("chapter", "Overview")
        current_section = section_data.get("section_title", "Overview")
        breadcrumb = section_data.get("breadcrumb", f"{chapter} > {current_section}")
        content_type = section_data.get("content_type", "narrative_text")
        section_anchor = section_data.get("anchor", "")

        # 1. Detect Figure Anchor
        fig_match = re.search(r"(?:^|\n)(?:#+\s*)?(Figure\s+\d+[-–]\d+:\s*[^\n]+)", raw_markdown, re.IGNORECASE)
        fig_anchor = fig_match.group(1).strip() if fig_match else ""
        if fig_anchor and content_type != "atomic_table":
            content_type = "figure_proxy"

        # 2. Check if Section is a Table
        is_table = (content_type == "atomic_table") or self.is_table_block(raw_markdown)
        table_meta = self.extract_table_and_footnotes(raw_markdown) if is_table else {}
        table_title = table_meta.get("title", "") or section_anchor
        if is_table:
            content_type = "atomic_table"

        anchor_name = table_title if is_table and table_title else (section_anchor or fig_anchor)

        final_chunks = []

        # Table Chunking
        if is_table:
            table_md = table_meta.get("table_md", raw_markdown)
            footnotes = table_meta.get("footnotes", "")
            if len(raw_markdown) <= self.max_chunk_chars:
                token_cnt = self.count_tokens(raw_markdown)
                prefix = self.build_breadcrumb_header(doc_id, doc_title, breadcrumb, page_num, "atomic_table", anchor_name)
                final_chunks.append({
                    "document_name": doc_name,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "page_number": page_num,
                    "section_title": current_section,
                    "breadcrumb": breadcrumb,
                    "content_type": "atomic_table",
                    "content": raw_markdown,
                    "embedding_content": f"{prefix}\n\n{raw_markdown}",
                    "token_count": token_cnt,
                    "metadata": {
                        "is_table": True,
                        "table_title": table_title,
                        "chapter": chapter,
                        "section": current_section,
                        "char_length": len(raw_markdown),
                    },
                })
            else:
                sub_chunks = self.split_table_safely(table_md, self.max_chunk_chars, title=table_title, footnotes=footnotes)
                for idx, sub_txt in enumerate(sub_chunks):
                    token_cnt = self.count_tokens(sub_txt)
                    prefix = self.build_breadcrumb_header(doc_id, doc_title, breadcrumb, page_num, "atomic_table", anchor_name)
                    final_chunks.append({
                        "document_name": doc_name,
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "page_number": page_num,
                        "section_title": current_section,
                        "breadcrumb": breadcrumb,
                        "content_type": "atomic_table",
                        "content": sub_txt,
                        "embedding_content": f"{prefix}\n\n{sub_txt}",
                        "token_count": token_cnt,
                        "metadata": {
                            "is_table": True,
                            "table_title": table_title,
                            "chapter": chapter,
                            "section": current_section,
                            "sub_chunk_index": idx,
                            "char_length": len(sub_txt),
                        },
                    })
        # Prose / Figure Proxy / Code Chunking
        else:
            if len(raw_markdown) <= self.max_chunk_chars:
                token_cnt = self.count_tokens(raw_markdown)
                prefix = self.build_breadcrumb_header(doc_id, doc_title, breadcrumb, page_num, content_type, anchor_name)
                final_chunks.append({
                    "document_name": doc_name,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "page_number": page_num,
                    "section_title": current_section,
                    "breadcrumb": breadcrumb,
                    "content_type": content_type,
                    "content": raw_markdown,
                    "embedding_content": f"{prefix}\n\n{raw_markdown}",
                    "token_count": token_cnt,
                    "metadata": {
                        "is_table": False,
                        "figure_anchor": fig_anchor,
                        "chapter": chapter,
                        "section": current_section,
                        "char_length": len(raw_markdown),
                    },
                })
            else:
                sub_chunks = self.split_prose_safely(raw_markdown, self.max_chunk_chars, self.overlap_chars)
                for idx, sub_txt in enumerate(sub_chunks):
                    token_cnt = self.count_tokens(sub_txt)
                    prefix = self.build_breadcrumb_header(doc_id, doc_title, breadcrumb, page_num, content_type, anchor_name)
                    final_chunks.append({
                        "document_name": doc_name,
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "page_number": page_num,
                        "section_title": current_section,
                        "breadcrumb": breadcrumb,
                        "content_type": content_type,
                        "content": sub_txt,
                        "embedding_content": f"{prefix}\n\n{sub_txt}",
                        "token_count": token_cnt,
                        "metadata": {
                            "is_table": False,
                            "figure_anchor": fig_anchor,
                            "chapter": chapter,
                            "section": current_section,
                            "sub_chunk_index": idx,
                            "char_length": len(sub_txt),
                        },
                    })

        return final_chunks

    def consolidate_micro_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate micro-chunks (< min_chunk_tokens) by merging them into adjacent chunks."""
        if not chunks:
            return []

        consolidated = []
        pending_prefix = ""

        for c in chunks:
            tok_len = c.get("token_count", self.count_tokens(c["content"]))
            content = c["content"]

            if pending_prefix:
                content = f"{pending_prefix}\n\n{content}"
                c["content"] = content
                c["token_count"] = self.count_tokens(content)
                prefix = self.build_breadcrumb_header(
                    c.get("doc_id", "DOC"),
                    c.get("doc_title", "Manual"),
                    c.get("breadcrumb", "Overview"),
                    c.get("page_number", 1),
                    c.get("content_type", "narrative_text"),
                    c.get("metadata", {}).get("figure_anchor", "")
                )
                c["embedding_content"] = f"{prefix}\n\n{content}"
                pending_prefix = ""

            # Check if this chunk is a micro-chunk
            if c["token_count"] < self.min_chunk_tokens and not c.get("metadata", {}).get("is_table", False):
                pending_prefix = content
            else:
                consolidated.append(c)

        # If last chunk was a micro-chunk, append to the last consolidated chunk
        if pending_prefix and consolidated:
            last = consolidated[-1]
            last_content = f"{last['content']}\n\n{pending_prefix}"
            last["content"] = last_content
            last["token_count"] = self.count_tokens(last_content)
            prefix = self.build_breadcrumb_header(
                last.get("doc_id", "DOC"),
                last.get("doc_title", "Manual"),
                last.get("breadcrumb", "Overview"),
                last.get("page_number", 1),
                last.get("content_type", "narrative_text"),
                last.get("metadata", {}).get("figure_anchor", "")
            )
            last["embedding_content"] = f"{prefix}\n\n{last_content}"
        elif pending_prefix and not consolidated and chunks:
            consolidated = chunks

        return consolidated
