-- Создание типов ENUM
CREATE TYPE chunk_granularity AS ENUM ('section', 'atomic');
CREATE TYPE chunk_type AS ENUM ('scope', 'requirement', 'definition', 'other');

-- Таблица документов
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'unknown',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Таблица секций
CREATE TABLE IF NOT EXISTS sections (
    section_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    part INT,
    subpart TEXT,
    section_number TEXT NOT NULL,
    section_title TEXT NOT NULL,
    anchor TEXT,
    page_start INT,
    page_end INT,
    reserved BOOLEAN NOT NULL DEFAULT false,
    source_kind TEXT NOT NULL DEFAULT 'toc',
    source_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Таблица чанков
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id TEXT NOT NULL REFERENCES sections(section_id) ON DELETE CASCADE,
    part INT,
    section_number TEXT NOT NULL,
    section_title TEXT NOT NULL,
    anchor TEXT,
    granularity chunk_granularity NOT NULL,
    parent_chunk_id TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    paragraph_path TEXT,
    chunk_kind chunk_type NOT NULL DEFAULT 'other',
    text_raw TEXT NOT NULL,
    embedding vector(4096),  -- Размерность для emb-qwen/qwen3-embedding-8b
    page_start INT,
    page_end INT,
    source_kind TEXT NOT NULL DEFAULT 'ocr_txt',
    source_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section_number ON chunks(section_number);
CREATE INDEX IF NOT EXISTS idx_chunks_granularity ON chunks(granularity);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_kind ON chunks(chunk_kind);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_chunk_id ON chunks(parent_chunk_id) WHERE parent_chunk_id IS NOT NULL;

-- Векторный индекс для поиска похожих чанков (HNSW - быстрый, но занимает больше места)
-- Создадим после загрузки данных, когда будет понятна размерность векторов
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- Staging таблицы для загрузки
CREATE TABLE IF NOT EXISTS staging_sections (
    row JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS staging_chunks (
    row JSONB NOT NULL
);
