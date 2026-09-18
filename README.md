# HydraCache

HydraCache is an enterprise hybrid RAG system with semantic caching designed to process SEC EDGAR 10-K filings.

## Problem Being Solved
Traditional RAG systems often retrieve the same context for identical or semantically similar queries, wasting computational resources and increasing latency. HydraCache solves this by implementing a Redis-based semantic cache that returns answers instantly for cache hits. For cache misses, it utilizes a sophisticated hybrid retrieval pipeline combining BM25 and Qdrant dense retrieval, fused via Reciprocal Rank Fusion (RRF), and re-ranked using a cross-encoder model before LLM synthesis.

## High-Level Architecture
1. **User Query**
2. **Redis Semantic Cache**: Checks for semantically similar previous queries.
    - **Cache Hit**: Returns cached answer immediately.
    - **Cache Miss**: Proceeds to retrieval.
3. **Hybrid Retrieval**:
    - BM25 (Sparse Retrieval)
    - Qdrant Dense Retrieval
4. **RRF Fusion**: Fuses results from both retrievers.
5. **Cross-Encoder Reranker**: Reranks the fused results to extract Top-5 Contexts.
6. **LLM Synthesis**: Synthesizes the answer using the context.
7. **Cache Write**: Writes the query and synthesized answer back to the semantic cache with metadata and citations.

## Tech Stack
- **Vector Database**: Qdrant
- **Semantic Cache & Metadata**: Redis Stack
- **Sparse Retrieval**: BM25
- **Embeddings & Reranking**: FastEmbed / Sentence-Transformers
- **Document Parsing**: pdfplumber, unstructured
- **API**: FastAPI
- **UI**: Streamlit
- **Evaluation**: Ragas

## Local Setup & Environment
1. Ensure you have Python 3.10+ installed.
2. Clone the repository.
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Copy the `.env.example` file to `.env` and fill in necessary details (API keys are not required for Phase 0).
   ```bash
   cp .env.example .env
   ```

## Docker Startup
HydraCache uses Docker to run local infrastructure (Qdrant and Redis).
1. Ensure Docker Desktop or Docker Engine is running.
2. Start the services:
   ```bash
   docker-compose up -d
   ```

## Verification Instructions
To verify that your environment and infrastructure are correctly configured:
```bash
python scripts/verify_infra.py
```
This script will check your Python version, and attempt to connect to Qdrant and Redis.

You can also run the basic test suite:
```bash
pytest tests/
```

## Requirements
Ensure you have a `.env` file in the root with:
```
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
GROQ_API_KEY=gsk_...
```

## Running the Application

### Streamlit Dashboard
You can run the Phase 4 UI dashboard to experiment with the system and view the evaluation analytics:
```bash
streamlit run app.py
```

### Automated Evaluation
To re-run the Ragas benchmarking and cache profiling:
```bash
# Set PYTHONPATH if needed depending on your shell
python eval/run_eval.py
python eval/run_cache_profile.py
```

## Phase 4 Evaluation Limitation

The evaluation framework, benchmark dataset, baseline, cache profiling,
failure analysis, and Streamlit dashboard are fully implemented.

The full Ragas quality evaluation was attempted against 40 evaluation
runs but was blocked by Groq evaluator API rate limits before any valid
Ragas scores were produced.

Therefore, no Ragas quality metric is reported as a final benchmark
result.

Cache performance was independently measured using the cache profiling
workload.

This limitation concerns the evaluation provider and does not indicate
a retrieval, caching, or synthesis failure in HydraCache.

## Phase 5: Production API

HydraCache is exposed as a production REST API using FastAPI.

### Start the Server
```bash
uvicorn src.api.main:app --port 8000
```

### Environment Configuration
The API relies on standard configuration via `.env` or environment variables:
- `API_KEY`: A secret string used for authenticating requests.
- `RATE_LIMIT_PER_MINUTE`: Defaults to 10.
- Standard Qdrant/Redis and LLM keys (`GROQ_API_KEY`).

### API Authentication & Rate Limiting
All core endpoints require the API Key to be passed via the `X-API-Key` header.
Rate limits (default 10 requests / minute) are tracked per unique API Key (using a secure SHA-256 hash identity in memory). 

### Endpoints
1. `GET /health` (Unauthenticated) - Basic liveness probe.
2. `POST /query` (Authenticated) - Submits a query to the full RAG+Cache pipeline. Returns `QueryResponse` (answer, citations, tokens, latency).
3. `POST /documents` (Authenticated) - Uploads a PDF (Max 50MB) and queues it for asynchronous Phase 1 ingestion. Returns a `job_id`.
4. `GET /documents/{job_id}` (Authenticated) - Polls the status of the background ingestion job.

### Example cURL
```bash
# Query
curl -X POST "http://localhost:8000/query" \
     -H "X-API-Key: your_secret_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"query": "What were Apple net sales in 2025?"}'

# Upload Document
curl -X POST "http://localhost:8000/documents" \
     -H "X-API-Key: your_secret_api_key_here" \
     -F "file=@data/raw/apple_10k_2025.pdf"
```

### Observability
All requests emit structured JSON logs. The raw API key is explicitly excluded from logs to prevent credential leakage. A unique `X-Request-ID` is assigned to each request and included in both the JSON logs and the response headers.

### Testing
- **Unit/Integration Tests**: `pytest tests/test_api.py -v` (runs endpoints via `TestClient`).
- **Live Server Manual Tests**: `python scripts/test_api.py` (executes realistic HTTP traffic and evaluates rate limit exhaustion).

### Known Limitations
- The `GET /documents/{job_id}` ingestion status relies on an in-memory dictionary. If the FastAPI process restarts, job histories are cleared. A distributed queue (e.g., Celery) would be required for durable job status tracking.

## Planned Phase Roadmap
- **[x] Phase 0**: Project Foundation & Environment Setup (Implemented)
- **[x] Phase 1**: Parsing, Chunking, BM25 + Qdrant Indexing (Implemented)
- **[x] Phase 2**: Hybrid Retrieval + RRF + Cross-Encoder Reranking (Implemented)
- **[x] Phase 3**: Semantic Caching (Implemented)
- **[x] Phase 4**: Evaluation + Benchmarking + Failure Analysis + Streamlit (Implemented)
- **[ ] Phase 5**: Production API + Auth + Rate Limiting + Observability (Planned)
- **[ ] Phase 6**: Multi-document + Multi-hop Retrieval (Planned)

## Phase 6: Multi-Document + Multi-Hop Retrieval
HydraCache supports multi-document reasoning via query decomposition and cross-document evidence merging. Queries comparing data points across time periods (e.g., Apple 10-K 2024 vs 2023) are automatically routed to a multi-hop retrieval pipeline. Discrepancies and restated figures are surfaced securely in the final synthesis rather than hallucinated or reconciled silently.
