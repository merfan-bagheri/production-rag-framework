import json
import logging
from typing import Any, Dict, List, Optional, Union
import psycopg2
from psycopg2.extras import Json, execute_batch, RealDictCursor
from pgvector.psycopg2 import register_vector

from rag_project.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    BASE_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PostgresClient:
    """PostgreSQL client abstraction with native pgvector, Multi-Document filtering, and Full-Text Search."""

    def __init__(
        self,
        host: str = DB_HOST,
        port: int = DB_PORT,
        dbname: str = DB_NAME,
        user: str = DB_USER,
        password: str = DB_PASSWORD,
    ):
        self.conn_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }
        self.conn = None

    def get_connection(self):
        """Get an active database connection and register pgvector extension."""
        if self.conn is None or self.conn.closed != 0:
            ports_to_try = [self.conn_params["port"]]
            alt_port = 5432 if self.conn_params["port"] != 5432 else 15432
            if alt_port not in ports_to_try:
                ports_to_try.append(alt_port)

            last_err = None
            for p in ports_to_try:
                try:
                    params = dict(self.conn_params)
                    params["port"] = p
                    self.conn = psycopg2.connect(**params)
                    self.conn_params["port"] = p
                    break
                except Exception as e:
                    last_err = e

            if self.conn is None or self.conn.closed != 0:
                raise last_err or Exception("Failed to connect to PostgreSQL database.")

            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.conn.commit()
            register_vector(self.conn)
        return self.conn


    def close(self):
        """Close connection cleanly."""
        if self.conn and self.conn.closed == 0:
            self.conn.close()

    def initialize_schema(self, schema_file: Optional[str] = None):
        """Execute the SQL schema migration file."""
        if schema_file is None:
            schema_file = str(BASE_DIR / "db" / "schema.sql")

        with open(schema_file, "r", encoding="utf-8") as f:
            sql_content = f.read()

        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute(sql_content)
        conn.commit()
        logger.info("Database schema initialized successfully with multi-doc isolation and compound indexes.")

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Batch insert chunks with multi-document metadata and embeddings."""
        if not chunks:
            return

        conn = self.get_connection()
        query = """
            INSERT INTO document_chunks (
                doc_id,
                doc_title,
                doc_category,
                document_name,
                page_number,
                section_title,
                breadcrumb,
                content_type,
                content,
                token_count,
                metadata,
                embedding
            ) VALUES (
                %(doc_id)s,
                %(doc_title)s,
                %(doc_category)s,
                %(document_name)s,
                %(page_number)s,
                %(section_title)s,
                %(breadcrumb)s,
                %(content_type)s,
                %(content)s,
                %(token_count)s,
                %(metadata)s,
                %(embedding)s
            )
        """

        def sanitize(val: Any) -> Any:
            if isinstance(val, str):
                return val.replace("\x00", "")
            if isinstance(val, dict):
                return {k: sanitize(v) for k, v in val.items()}
            if isinstance(val, list):
                return [sanitize(v) for v in val]
            return val

        formatted_records = []
        for c in chunks:
            meta = sanitize(c.get("metadata", {}))
            formatted_records.append({
                "doc_id": sanitize(c.get("doc_id", "unknown")),
                "doc_title": sanitize(c.get("doc_title", "unknown")),
                "doc_category": sanitize(c.get("doc_category", "FPGA_IP_GUIDE")),
                "document_name": sanitize(c.get("document_name", "doc.pdf")),
                "page_number": c.get("page_number", 1),
                "section_title": sanitize(c.get("section_title", "General")),
                "breadcrumb": sanitize(c.get("breadcrumb", "")),
                "content_type": sanitize(c.get("content_type", "prose")),
                "content": sanitize(c.get("content", "")),
                "token_count": c.get("token_count", 0),
                "metadata": Json(meta),
                "embedding": c.get("embedding", None),
            })

        with conn.cursor() as cur:
            execute_batch(cur, query, formatted_records, page_size=batch_size)
        conn.commit()
        logger.info(f"Inserted {len(chunks)} multi-doc chunks into document_chunks table.")

    def hybrid_search_rrf(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k_dense: int = 40,
        top_k_sparse: int = 40,
        rrf_k: int = 60,
        limit: int = 20,
        doc_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector + sparse full-text search in a single optimized CTE roundtrip with native HNSW index acceleration."""
        conn = self.get_connection()
        
        target_docs = None
        if doc_filter:
            if isinstance(doc_filter, str):
                target_docs = [doc_filter]
            elif isinstance(doc_filter, list) and len(doc_filter) > 0:
                target_docs = doc_filter

        # Construct precise WHERE conditions to ensure PostgreSQL uses HNSW index
        if target_docs:
            dense_where = "WHERE doc_id = ANY(%(doc_ids)s)"
            sparse_where = "WHERE doc_id = ANY(%(doc_ids)s) AND tsv_content @@ plainto_tsquery('english', %(qtext)s)"
        else:
            dense_where = ""
            sparse_where = "WHERE tsv_content @@ plainto_tsquery('english', %(qtext)s)"

        query = f"""
            WITH dense_search AS (
                SELECT id, 1 - (embedding <=> %(qemb)s::vector) AS dense_similarity,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %(qemb)s::vector) AS dense_rank
                FROM document_chunks
                {dense_where}
                ORDER BY embedding <=> %(qemb)s::vector
                LIMIT %(k_dense)s
            ),
            sparse_search AS (
                SELECT id, ts_rank_cd(tsv_content, plainto_tsquery('english', %(qtext)s)) AS sparse_score,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv_content, plainto_tsquery('english', %(qtext)s)) DESC) AS sparse_rank
                FROM document_chunks
                {sparse_where}
                LIMIT %(k_sparse)s
            )
            SELECT dc.id,
                   dc.doc_id,
                   dc.doc_title,
                   dc.doc_category,
                   dc.document_name,
                   dc.page_number,
                   dc.section_title,
                   dc.breadcrumb,
                   dc.content_type,
                   dc.content,
                   dc.token_count,
                   dc.metadata,
                   ds.dense_similarity,
                   ds.dense_rank,
                   ss.sparse_rank,
                   COALESCE(1.0 / (%(rrf_k)s + ds.dense_rank), 0.0) + COALESCE(1.0 / (%(rrf_k)s + ss.sparse_rank), 0.0) AS rrf_score
            FROM (
                SELECT id FROM dense_search
                UNION
                SELECT id FROM sparse_search
            ) u
            JOIN document_chunks dc ON dc.id = u.id
            LEFT JOIN dense_search ds ON ds.id = u.id
            LEFT JOIN sparse_search ss ON ss.id = u.id
            ORDER BY rrf_score DESC
            LIMIT %(limit)s;
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                query,
                {
                    "qemb": query_embedding,
                    "qtext": query_text,
                    "doc_ids": target_docs,
                    "k_dense": top_k_dense,
                    "k_sparse": top_k_sparse,
                    "rrf_k": rrf_k,
                    "limit": limit
                }
            )
            results = cur.fetchall()
        return [dict(r) for r in results]

    def get_chunks_by_pages(
        self,
        pages: List[int],
        doc_id: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all chunks belonging to specific page numbers, optionally filtered by doc_id or list of doc_ids."""
        if not pages:
            return []
        conn = self.get_connection()
        if doc_id:
            if isinstance(doc_id, str):
                target_docs = [doc_id]
            else:
                target_docs = list(doc_id)
            query = """
                SELECT id, doc_id, doc_title, doc_category, document_name, page_number, section_title, breadcrumb, content_type, content, metadata
                FROM document_chunks
                WHERE page_number = ANY(%s) AND doc_id = ANY(%s)
                ORDER BY doc_id ASC, page_number ASC, id ASC;
            """
            params = (pages, target_docs)
        else:
            query = """
                SELECT id, doc_id, doc_title, doc_category, document_name, page_number, section_title, breadcrumb, content_type, content, metadata
                FROM document_chunks
                WHERE page_number = ANY(%s)
                ORDER BY doc_id ASC, page_number ASC, id ASC;
            """
            params = (pages,)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        return [dict(r) for r in results]


    def get_chunk_count(self) -> int:
        """Return the total number of chunks stored in the database."""
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            count = cur.fetchone()[0]
        return count

    def get_doc_page_stats(self) -> List[Dict[str, Any]]:
        """Return page and chunk statistics per document."""
        conn = self.get_connection()
        query = """
            SELECT doc_id, doc_title, doc_category, COUNT(*) as chunk_count, MAX(page_number) as max_page, MIN(page_number) as min_page
            FROM document_chunks
            GROUP BY doc_id, doc_title, doc_category
            ORDER BY doc_id ASC;
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            results = cur.fetchall()
        return [dict(r) for r in results]
