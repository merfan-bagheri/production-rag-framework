import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pymupdf
from sentence_transformers import SentenceTransformer
import torch

from rag_project.config import (
    DOCS_DIR,
    DEFAULT_PDF_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    GOOGLE_API_KEY_FILE,
    DOCUMENT_METADATA_MAP
)
from rag_project.db.postgres_client import PostgresClient
from rag_project.ingestion.pdf_parser import PDFParser
from rag_project.ingestion.chunker import StructureAwareChunker
from rag_project.retrieval.doc_router import DOCUMENT_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiDocBatchIngestor:
    """Batch Multi-Document Parser, Hierarchical Chunker & Vector Ingestor."""

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        db_client: Optional[PostgresClient] = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.docs_dir = docs_dir or DOCS_DIR
        self.db = db_client or PostgresClient()
        self.chunker = StructureAwareChunker()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        self.embedder.max_seq_length = 512
        if hasattr(self.embedder, "tokenizer") and self.embedder.tokenizer is not None:
            self.embedder.tokenizer.model_max_length = 512

    def discover_pdf_files(self) -> List[Dict[str, Any]]:
        """Scan docs directory and root for all PDF manuals."""
        pdf_paths = []
        if self.docs_dir.exists():
            pdf_paths.extend(list(self.docs_dir.glob("*.pdf")))
        if DEFAULT_PDF_PATH.exists() and DEFAULT_PDF_PATH not in pdf_paths:
            # If doc.pdf in root is identical to pg063-dist-mem-gen.pdf, avoid duplicating
            has_pg063 = any(p.name == "pg063-dist-mem-gen.pdf" for p in pdf_paths)
            if not has_pg063:
                pdf_paths.append(DEFAULT_PDF_PATH)

        pdf_info_list = []
        for path in pdf_paths:
            filename = path.name.lower()
            meta = DOCUMENT_METADATA_MAP.get(filename, {
                "doc_id": path.stem.upper(),
                "doc_title": f"Xilinx Manual {path.stem.upper()}",
                "doc_category": "FPGA_DOCUMENTATION"
            })

            doc = pymupdf.open(str(path))
            total_pages = len(doc)
            doc.close()

            pdf_info_list.append({
                "path": path,
                "filename": path.name,
                "doc_id": meta["doc_id"],
                "doc_title": meta["doc_title"],
                "doc_category": meta["doc_category"],
                "total_pages": total_pages
            })

        return pdf_info_list

    def process_document(self, pdf_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract structured sections and generate contextual chunks for a single document."""
        path = pdf_info["path"]
        doc_id = pdf_info["doc_id"]
        doc_title = pdf_info["doc_title"]
        doc_category = pdf_info["doc_category"]

        logger.info(f"Processing '{pdf_info['filename']}' ({doc_id}) - {pdf_info['total_pages']} pages...")
        parser = PDFParser(str(path))
        sections = parser.extract_page_by_page()

        raw_chunks = []
        for s in sections:
            s_chunks = self.chunker.chunk_section(s, doc_id=doc_id, doc_title=doc_title)
            for c in s_chunks:
                c["doc_id"] = doc_id
                c["doc_title"] = doc_title
                c["doc_category"] = doc_category
            raw_chunks.extend(s_chunks)

        # Consolidate micro-chunks
        doc_chunks = self.chunker.consolidate_micro_chunks(raw_chunks)
        logger.info(f"Generated {len(doc_chunks)} consolidated chunks for {doc_id}.")
        return doc_chunks

    def run(self, reinit_schema: bool = True) -> Dict[str, Any]:
        """Execute full batch ingestion across all discovered documents with coverage assertions."""
        start_time = time.time()
        logger.info("=== STARTING MULTI-DOCUMENT BATCH INGESTION PIPELINE ===")

        # 1. Discover PDFs
        pdf_files = self.discover_pdf_files()
        logger.info(f"Discovered {len(pdf_files)} PDF manuals in repository.")

        # 2. Reinitialize Schema
        if reinit_schema:
            logger.info("Reinitializing PostgreSQL schema with multi-document isolation...")
            self.db.initialize_schema()

        # 3. Extract & Chunk All Documents
        all_chunks: List[Dict[str, Any]] = []
        for pdf in pdf_files:
            chunks = self.process_document(pdf)
            all_chunks.extend(chunks)

        logger.info(f"Total multi-document chunks created across all manuals: {len(all_chunks)}")

        # 4. Dense Vector Embeddings in Batches
        logger.info("Computing dense embeddings across complete multi-document corpus...")
        texts_to_embed = [c.get("embedding_content", c["content"]) for c in all_chunks]
        embeddings = self.embedder.encode(
            texts_to_embed,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb.tolist()

        # 5. Batch Ingestion into PostgreSQL
        logger.info("Batch inserting records into PostgreSQL document_chunks table...")
        self.db.insert_chunks_batch(all_chunks, batch_size=200)

        # 6. Post-Ingestion Assertions & Coverage Verification
        stats = self.db.get_doc_page_stats()
        logger.info("=== POST-INGESTION COVERAGE REPORT ===")
        print("\n" + "=" * 90)
        print("MULTI-DOCUMENT DATABASE COVERAGE REPORT")
        print("=" * 90)
        print(f"{'Doc ID':<10} | {'Document Title':<42} | {'Pages':<10} | {'Chunks':<8}")
        print("-" * 90)

        stats_map = {s["doc_id"]: s for s in stats}
        for pdf in pdf_files:
            doc_id = pdf["doc_id"]
            db_stat = stats_map.get(doc_id)
            if db_stat:
                max_page = db_stat["max_page"]
                chunk_cnt = db_stat["chunk_count"]
                print(f"{doc_id:<10} | {pdf['doc_title'][:42]:<42} | 1..{max_page:<6} | {chunk_cnt:<8}")
                assert max_page >= pdf["total_pages"] - 6, (
                    f"Coverage assertion failed for {doc_id}: expected ~{pdf['total_pages']} pages, got {max_page}"
                )
            else:
                logger.error(f"Missing doc_id {doc_id} in database stats!")

        total_records = self.db.get_chunk_count()
        duration = round(time.time() - start_time, 2)
        print("=" * 90)
        print(f"Total Documents: {len(pdf_files)} | Total Chunks: {total_records} | Ingestion Time: {duration}s")
        print("=" * 90 + "\n")

        return {
            "total_docs": len(pdf_files),
            "total_chunks": total_records,
            "duration_s": duration,
            "stats": stats
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Document Batch Ingestor")
    parser.add_argument("--no-reinit", dest="reinit", action="store_false", help="Do not drop and reinitialize table")
    args = parser.parse_args()

    ingestor = MultiDocBatchIngestor()
    ingestor.run(reinit_schema=args.reinit)
