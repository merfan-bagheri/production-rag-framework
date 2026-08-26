import copy
import logging
import re
from typing import Any, Dict, List, Optional, Set
from rag_project.db.postgres_client import PostgresClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NeighborExpander:
    """Dynamic Neighbor Lookahead Expander.
    
    Resolves Context Fragmentation and Boundary Truncation at retrieval time.
    For high-confidence chunks (Score >= min_score or Top-3), checks if subsequent
    sequential chunks (C_{i+1}) from the same document and section exist in PostgreSQL,
    and dynamically stitches or appends them into the retrieval context.
    """

    def __init__(
        self,
        db_client: Optional[PostgresClient] = None,
        min_score: float = 0.08,
        max_expansions: int = 4
    ):
        self.db = db_client or PostgresClient()
        self.min_score = min_score
        self.max_expansions = max_expansions

    def is_boundary_unclosed(self, text: str) -> bool:
        """Check if text ends without terminal punctuation, on a comma, or inside an unclosed table/list."""
        stripped = text.strip()
        if not stripped:
            return False

        # Ends with comma, semicolon, dash, or open parenthesis
        if bool(re.search(r"[,;\-\(]\s*$", stripped)):
            return True

        # Ends with unclosed table row (starts with | but doesn't end with |)
        lines = [l.strip() for l in stripped.split("\n") if l.strip()]
        if lines and lines[-1].startswith("|") and not lines[-1].endswith("|"):
            return True

        # Trailing section header without body
        if lines and bool(re.match(r"^#{1,6}\s+", lines[-1])):
            return True

        # Does not end with terminal punctuation (., !, ?, :, ```, |)
        if not bool(re.search(r"[\.!\?:\`\|\"]\s*$", stripped)):
            return True

        return False

    def fetch_adjacent_chunks(self, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch chunk metadata and content from PostgreSQL by ID."""
        if not chunk_ids:
            return {}

        conn = self.db.get_connection()
        query = """
            SELECT id, document_name, doc_id, doc_title, page_number,
                   section_title, breadcrumb, content_type, content,
                   token_count, metadata
            FROM document_chunks
            WHERE id = ANY(%s)
        """
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (list(chunk_ids),))
            records = cur.fetchall()

        result_map = {}
        for r in records:
            result_map[r["id"]] = {
                "id": r["id"],
                "document_name": r["document_name"],
                "doc_id": r["doc_id"],
                "doc_title": r["doc_title"],
                "page_number": r["page_number"],
                "section_title": r["section_title"],
                "breadcrumb": r["breadcrumb"],
                "content_type": r["content_type"],
                "content": r["content"],
                "token_count": r["token_count"],
                "metadata": r.get("metadata", {}),
                "is_expanded_neighbor": True
            }
        return result_map

    def expand_neighbors(
        self,
        ranked_chunks: List[Dict[str, Any]],
        target_docs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Expand high-confidence or boundary-unclosed chunks with their sequential neighbors."""
        if not ranked_chunks:
            return []

        expanded_results = []
        seen_ids: Set[int] = {c.get("id") for c in ranked_chunks if c.get("id") is not None}
        expansion_candidates: List[int] = []

        # Step 1: Identify chunks that warrant sequential lookahead expansion & multi-part table completion
        for idx, chunk in enumerate(ranked_chunks):
            c_id = chunk.get("id")
            if c_id is None:
                continue

            content = chunk.get("content", "")
            score = chunk.get("rerank_score", chunk.get("rrf_score", 0.0))
            is_top_rank = (idx < 5)
            is_high_confidence = (score >= self.min_score)
            is_unclosed = self.is_boundary_unclosed(content)
            is_table = (chunk.get("content_type") == "atomic_table" or "Table " in content or "|" in content)

            # Lookahead trigger: Multi-part table, unclosed boundary, high score, or top rank
            if (is_top_rank or is_high_confidence or is_unclosed or is_table) and len(expansion_candidates) < self.max_expansions * 4:
                max_chain = 4 if is_table else (2 if is_top_rank else 1)
                for step in range(1, max_chain + 1):
                    next_id = c_id + step
                    if next_id not in seen_ids and next_id not in expansion_candidates:
                        expansion_candidates.append(next_id)

        # Step 2: Batch fetch neighbor records from DB
        neighbor_map = self.fetch_adjacent_chunks(expansion_candidates)

        # Step 3: Stitch contiguous same-document neighbors directly into parent chunks or append as sources
        for chunk in ranked_chunks:
            c_id = chunk.get("id")
            if c_id is None:
                expanded_results.append(chunk)
                continue

            curr_text = chunk.get("content", "").rstrip()
            parent_doc = chunk.get("doc_id")

            # Check consecutive next chunks (c_id + 1, c_id + 2, etc.)
            step = 1
            while (c_id + step) in neighbor_map:
                next_chunk = neighbor_map[c_id + step]
                if next_chunk.get("doc_id") != parent_doc:
                    break

                child_text = next_chunk.get("content", "").strip()
                is_table_cont = (
                    "Cont’d" in child_text or "Cont'd" in child_text or
                    (chunk.get("content_type") == "atomic_table" and next_chunk.get("content_type") == "atomic_table")
                )

                # Strip redundant header lines if child repeats breadcrumb heading
                if child_text.startswith("#"):
                    lines = child_text.split("\n")
                    if len(lines) > 1 and lines[0].startswith("#"):
                        child_text = "\n".join(lines[1:]).strip()

                curr_text = f"{curr_text}\n\n{child_text}"
                chunk["end_page"] = next_chunk.get("page_number", chunk.get("page_number"))
                seen_ids.add(next_chunk["id"])
                logger.debug(f"Stitched continuation chunk ID {next_chunk['id']} into parent ID {c_id}.")
                step += 1

            chunk["content"] = curr_text
            expanded_results.append(chunk)

        return expanded_results
