import argparse
import logging
import time
from typing import Optional
from sentence_transformers import SentenceTransformer
import torch

from rag_project.config import (
    DEFAULT_PDF_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    GOOGLE_API_KEY_FILE
)
from rag_project.db.postgres_client import PostgresClient
from rag_project.ingestion.pdf_parser import PDFParser
from rag_project.ingestion.chunker import StructureAwareChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class IngestionPipeline:
    """End-to-end high-recall ingestion pipeline:
    Hierarchical TOC Extraction -> Leaf Section Preservation -> Contextual Prefixing -> Dense Embeddings -> PostgreSQL.
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        db_client: Optional[PostgresClient] = None,
    ):
        self.pdf_path = pdf_path or str(DEFAULT_PDF_PATH)
        self.parser = PDFParser(self.pdf_path)
        self.chunker = StructureAwareChunker()
        self.db = db_client or PostgresClient()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        self.embedder.max_seq_length = 512
        if hasattr(self.embedder, "tokenizer") and self.embedder.tokenizer is not None:
            self.embedder.tokenizer.model_max_length = 512

    def run(self, reinit_schema: bool = True, enrich_with_gemini: bool = False) -> int:
        """Run the full high-recall ingestion pipeline."""
        start_time = time.time()
        logger.info(f"Starting high-recall ingestion for document: {self.pdf_path}")

        # 1. Initialize DB Schema with Weighted Full-Text Search
        if reinit_schema:
            logger.info("Initializing PostgreSQL schema and vector extension...")
            self.db.initialize_schema()

        # 2. Parse PDF with hierarchical TOC and leaf preservation
        logger.info("Parsing PDF with hierarchical TOC and leaf section extraction...")
        sections = self.parser.extract_page_by_page()
        logger.info(f"Extracted {len(sections)} structured sections from document.")

        # 3. Structure-aware chunking with contextual prefixes
        logger.info("Generating structure-aware chunks with contextual prefixes...")
        all_chunks = []
        for s in sections:
            chunks = self.chunker.chunk_section(s)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks created: {len(all_chunks)}")

        # Optional: Enrich with Gemini API if requested
        if enrich_with_gemini and GOOGLE_API_KEY_FILE.exists():
            try:
                from rag_project.ingestion.gemini_enricher import GeminiDocumentEnricher
                enricher = GeminiDocumentEnricher(api_key_file=GOOGLE_API_KEY_FILE)
                all_chunks = enricher.enrich_all_chunks(all_chunks)
            except Exception as e:
                logger.warning(f"Gemini enrichment failed ({e}), continuing with standard chunks.")

        # 4. Dense Vector Embeddings on Contextual Prefixed Text
        logger.info("Computing dense embeddings on contextual prefixed content...")
        texts_to_embed = [c.get("embedding_content", c["content"]) for c in all_chunks]
        embeddings = self.embedder.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb.tolist()

        # 5. Batch Ingestion into PostgreSQL
        logger.info("Batch inserting records into PostgreSQL...")
        self.db.insert_chunks_batch(all_chunks, batch_size=100)

        # 6. Verification
        count = self.db.get_chunk_count()
        duration = round(time.time() - start_time, 2)
        logger.info(f"Ingestion completed in {duration}s. Total stored records in database: {count}")
        return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xilinx LogiCORE IP RAG Ingestion Pipeline")
    parser.add_argument("--enrich", action="store_true", help="Enable Gemini API chunk enrichment")
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    pipeline.run(enrich_with_gemini=args.enrich)
