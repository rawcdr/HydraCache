# HydraCache Execution Flow

This file documents how the code actually executes and interacts.

## Phase 0

### Entry Point
- `scripts/verify_infra.py`
- `tests/test_infra.py`

### Execution Flow
1. **Verification Script**: User runs `python scripts/verify_infra.py`.
2. Script verifies current Python version.
3. Script attempts a connection to Qdrant (HTTP) on `localhost:6333`.
4. Script attempts a connection to Redis (TCP) on `localhost:6379`.
5. Script outputs success or failure for each check.

### Function Call Chain
```text
scripts/verify_infra.py
 -> check_python_version()
 -> check_qdrant()
 -> check_redis()
```

### Data Flow
N/A (No data ingestion yet)

### External Dependencies
- Qdrant (Docker: `qdrant/qdrant:latest`)
- Redis Stack (Docker: `redis/redis-stack:latest`)

### Files Changed
- `scripts/verify_infra.py`
- `tests/test_infra.py`
- `docker-compose.yml`
- configuration boilerplate (`.env`, `requirements.txt`, etc.)

### Phase Changes
- Created foundational directory structure.
- Created `docker-compose.yml` for Qdrant and Redis.
- Created `verify_infra.py` script.
- Created basic tests for infrastructure.

## Phase 1 (Indexing Pipeline)

### Entry Point
- `scripts/run_indexing.py`

### Execution Flow
1. **Parser**: Iterates over PDF pages using `pdfplumber`. Extracts tables and converts them to markdown. Extracts remaining text. Emits logical `Document` objects with `chunk_type` metadata.
2. **Chunker**: 
    - Tables: Yielded directly as Parent=Child chunks.
    - Text: Split into sentences, then grouped into sliding windows (size 3, overlap 1) to create Child chunks. Parent text is retained in metadata.
3. **Dense Indexing**: FastEmbed `bge-small-en-v1.5` embeds the Child chunks. They are upserted into Qdrant (`apple_10k` collection), with payload preserving `parent_id` and `chunk_type`.
4. **Sparse Indexing**: Tokenizes Child chunks and builds a `BM25Okapi` index, saving it to disk (`data/processed/bm25_index.pkl`).

### Function Call Chain
```text
scripts/run_indexing.py
 -> PDFParser.parse()
 -> TextChunker.chunk_documents()
 -> DenseIndexer.index()
 -> SparseIndexer.index()
```

## Phase 2 (Retrieval Pipeline)

### Milestone 2: Dense Retrieval
1. **DenseRetriever**: `src.retrieval.dense.DenseRetriever.retrieve()` is called with a query and `top_k`.
2. Uses `QdrantClient` configured with the Phase 1 embedding model (`BAAI/bge-small-en-v1.5`).
3. Executes `client.query()` to automatically embed the query text and perform a vector search in Qdrant.
4. Converts Qdrant response payload into `RetrievalResult` objects preserving chunk_id, document text, score, and parent_id.

### Function Call Chain
```text
scripts/test_dense.py
 -> DenseRetriever.retrieve()
 -> QdrantClient.query()
 -> DenseRetriever.to_retrieval_result()
```
