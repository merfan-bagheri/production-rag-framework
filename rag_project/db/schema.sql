-- PostgreSQL + pgvector Schema for Multi-Document Xilinx Knowledge Engine

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Drop existing table if needed during re-initialization
DROP TABLE IF EXISTS document_chunks CASCADE;

-- 3. Document chunks table with Multi-Document Isolation & Metadata
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(64) NOT NULL DEFAULT 'unknown',
    doc_title VARCHAR(255) NOT NULL DEFAULT 'unknown',
    doc_category VARCHAR(100) DEFAULT 'FPGA_IP_GUIDE',
    document_name TEXT NOT NULL DEFAULT 'doc.pdf',
    page_number INTEGER NOT NULL,
    section_title TEXT,
    breadcrumb TEXT,
    content_type TEXT NOT NULL DEFAULT 'prose', -- 'table', 'prose', 'register_spec', 'timing'
    content TEXT NOT NULL,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384),
    tsv_content tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', 'Doc: ' || doc_id || ' ' || doc_title || ' Page ' || page_number || ' ' || COALESCE(section_title, '') || ' ' || COALESCE(breadcrumb, '')), 'A') ||
        setweight(to_tsvector('english', content), 'B')
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. HNSW Index for fast approximate nearest neighbor cosine distance vector search
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_embedding 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 5. GIN Index for Weighted Full-Text Search (tsvector)
CREATE INDEX IF NOT EXISTS idx_chunks_gin_tsv 
ON document_chunks 
USING gin (tsv_content);

-- 6. Compound & Filter Indexes for Multi-Document Isolation & Speed
CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON document_chunks (doc_id, page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_category ON document_chunks (doc_category);
