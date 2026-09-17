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

## Planned Phase Roadmap
- **[x] Phase 0**: Project Foundation & Environment Setup (Implemented)
- **[x] Phase 1**: Parsing, Chunking, BM25 + Qdrant Indexing (Implemented)
- **[x] Phase 2**: Hybrid Retrieval + RRF + Cross-Encoder Reranking (In Progress - Milestone 2 Complete)
- **[ ] Phase 3**: Semantic Caching (Planned)
- **[ ] Phase 4**: Evaluation + Benchmarking + Failure Analysis + Streamlit (Planned)
- **[ ] Phase 5**: Production API + Auth + Rate Limiting + Observability (Planned)
- **[ ] Phase 6**: Multi-document + Multi-hop Retrieval (Planned)
