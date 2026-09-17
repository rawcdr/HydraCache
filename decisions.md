# HydraCache Engineering Decisions

This file is a chronological engineering decision log.

## [2026-09-16 21:42]
### Decision: Define Phase 0 project structure, environment, and dependencies.
### Context: Initial setup of the HydraCache hybrid RAG system. We need a clean project root, dependency management, and local infrastructure setup before implementing Phase 1 functionality.
### Decision Made: 
- Create a standard Python data project structure (`src`, `data`, `notebooks`, `scripts`, etc.).
- Use a `requirements.txt` file with pinned versions for reproducibility.
- Setup `docker-compose.yml` with Redis Stack and Qdrant.
- Defer all implementation (parsing, BM25, Qdrant indexing, endpoints) to Phase 1.
### Why: To establish a stable foundation and reproducibility before writing core application logic, following the Phase 0 requirements.
### Alternatives Considered: 
- Using `poetry` or `pipenv` for dependency management.
### Why Alternatives Were Rejected: `requirements.txt` is simpler and explicitly requested for this setup unless conflicts arise. Pinned versions provide the same level of reproducibility.
### Impact: Establishes the boilerplate code, basic tests, and infrastructure verification.
### Phase: Phase 0

## [2026-09-16 21:43]
### Decision: Upgrade fastembed to 0.8.0 and use `docker compose` instead of `docker-compose`.
### Context: Initial setup failed because `fastembed==0.2.5` is incompatible with the system Python version (>=3.13 not supported in 0.2.5), and `docker-compose` is deprecated in favor of `docker compose` plugin.
### Decision Made: Update `requirements.txt` to pin `fastembed==0.8.0` and run docker using the newer `docker compose` syntax.
### Why: To ensure compatibility with the host system's modern Python and Docker setups.
### Alternatives Considered: Downgrading Python.
### Why Alternatives Were Rejected: Unnecessary friction and goes against using modern Python 3.10+ as requested.
### Impact: Resolves installation and container startup blockers.
### Phase: Phase 0

## [2026-09-16 21:44]
### Decision: Upgrade unstructured to 0.27.6 and defer Docker execution to user.
### Context: `unstructured==0.12.6` is incompatible with Python >=3.12. Furthermore, Docker is not available in the current environment's PATH to spin up Qdrant and Redis automatically.
### Decision Made: Update `unstructured==0.27.6`. Defer the actual execution of `docker compose up -d` and integration tests to the user's manual verification step.
### Why: To unblock dependency installation, and acknowledge environmental constraints.
### Alternatives Considered: N/A.
### Why Alternatives Were Rejected: Docker cannot be installed mid-flight in this restricted environment.
### Impact: Python environment is successfully setup, but infrastructure must be verified manually by the user.
### Phase: Phase 0

## [2026-09-16 21:46]
### Decision: Unpin `numpy` and `pandas` versions.
### Context: Installation failed for `numpy==1.26.4` because pip attempted to build from source (no Python 3.13 wheel available for that old version) and the Windows host lacks a C/C++ compiler.
### Decision Made: Change `numpy==1.26.4` to `numpy` and `pandas==2.2.1` to `pandas` in `requirements.txt` to allow fetching the latest compatible pre-built wheels.
### Why: To ensure a successful installation without requiring the user to install MSVC build tools.
### Alternatives Considered: Instructing user to install Visual Studio Build Tools.
### Why Alternatives Were Rejected: Unnecessary friction for Phase 0 setup. Modern wheels exist for newer versions.
### Impact: Relies on newer major versions (e.g., numpy 2.x) which should be compatible with our planned usage, but might require minor syntax adjustments later.
### Phase: Phase 0

## [2026-09-16 21:47]
### Decision: Unpin `requests` version.
### Context: Conflicting dependencies detected during pip installation. `requests==2.31.0` conflicted with `unstructured==0.27.6` which requires `requests>=2.32.5`.
### Decision Made: Change `requests==2.31.0` to `requests` in `requirements.txt`.
### Why: To allow `pip` to automatically resolve and install a compatible `requests` version.
### Alternatives Considered: Downgrading unstructured further.
### Why Alternatives Were Rejected: Unstructured needs to remain somewhat current for Python 3.13 compatibility; unpinning requests is much safer.
### Impact: Resolves dependency conflict during setup.
### Phase: Phase 0

## [2026-09-16 21:48]
### Decision: Unpin `streamlit` and `ragas`.
### Context: `streamlit==1.32.2` enforces `numpy<2`, but `fastembed==0.8.0` requires `numpy>=2.1.0` on Python 3.13, creating a resolution conflict.
### Decision Made: Change `streamlit==1.32.2` to `streamlit` and `ragas==0.1.5` to `ragas` in `requirements.txt`.
### Why: To allow `pip` to fetch a newer version of streamlit that supports numpy 2.x.
### Alternatives Considered: Pinning numpy to <2, which would break fastembed on Python 3.13.
### Why Alternatives Were Rejected: We need both UI (streamlit) and embeddings (fastembed) to work on the modern host Python version.
### Impact: Resolves the final dependency resolution conflict.
### Phase: Phase 0

## [2026-09-16 23:08]
### Decision: Switch virtual environment to Python 3.10 and revert to originally pinned requirements.
### Context: Python 3.13 caused unresolvable dependency conflicts and required C++ build tools for `numpy` and `unstructured`. A local installation of Python 3.10 was found on the system.
### Decision Made: Destroyed the Python 3.13 `.venv` and recreated it using `py -3.10 -m venv .venv`. Reverted `requirements.txt` to the exact original pinned versions (e.g. `fastembed==0.2.5`, `unstructured==0.12.6`, `numpy==1.26.4`).
### Why: Python 3.10 has stable, pre-compiled wheels for all the required data science packages. This perfectly matches the Phase 0 specification and avoids compilation errors.
### Alternatives Considered: Installing Visual Studio Build tools and maintaining loose version bounds.
### Why Alternatives Were Rejected: Not feasible to install heavy system dependencies automatically, and strict pinning guarantees reproducibility as requested.
### Impact: Python dependencies successfully installed. Docker setup remains deferred.
### Phase: Phase 0

## [2026-09-17 21:30]
### Decision: Refactor Phase 1 Pipeline for Parent-Child / Sentence-Window chunking and explicit Table Extraction.
### Context: The initial Phase 1 pipeline used LangChain's basic `PDFPlumberLoader` and `RecursiveCharacterTextSplitter`. An architectural audit revealed this failed to isolate tables, destroyed context, and lacked parent-child linking for advanced RAG retrieval.
### Decision Made: 
- Dropped `PDFPlumberLoader` in favor of a custom `pdfplumber` loop that isolates `Table` objects (formatting them as Markdown) and extracts remaining text as `Text`.
- Replaced character-splitting with a custom Parent-Child sentence-window chunking strategy (3-sentence windows overlapping by 1).
- Updated Qdrant index to clear existing collections before upserting, to avoid duplicate vector issues during iterative testing, and populated payloads with rich metadata (`parent_id`, `chunk_type`).
### Why: To satisfy the strict HydraCache specifications for high-precision retrieval while preserving document structure.
### Alternatives Considered: Using Unstructured's `partition_pdf`.
### Why Alternatives Were Rejected: Unstructured with `infer_table_structure` requires `tesseract` and `pdf2image`, which carry heavy system dependencies that complicate Windows installations. Custom `pdfplumber` logic is lighter and robust enough.
### Impact: Chunk counts increased from 363 to 753. Tables are now properly preserved in markdown format. Vectors include contextual payload properties.
### Phase: Phase 1 (Audit Correction)

## [2026-09-17 23:48]
### Decision: Upgrade `fastembed` to 0.3.4 for Reranking support.
### Context: Phase 2 requires a local Cross-Encoder reranker. The previously pinned `fastembed==0.2.5` only supported dense embeddings.
### Decision Made: Upgraded to `fastembed==0.3.4` in `requirements.txt`.
### Why: To use the native `TextCrossEncoder` provided by FastEmbed for `BAAI/bge-reranker-base` without needing a massive PyTorch dependency tree via `sentence-transformers`.
### Alternatives Considered: Using `sentence-transformers`.
### Why Alternatives Were Rejected: Bloats the environment heavily. FastEmbed maintains the lightweight, ONNX-based standard we chose in Phase 0.
### Impact: Enables Cross-Encoder reranking locally; `qdrant-client` remains compatible.
### Phase: Phase 2 (Milestone 1)

## [2026-09-18 00:36]
### Decision: Dense Retrieval using native QdrantClient FastEmbed Integration.
### Context: Phase 2 Milestone 2 requires querying the existing Qdrant collection with `bge-small-en-v1.5`.
### Decision Made: Used `QdrantClient.set_model("BAAI/bge-small-en-v1.5")` and `client.query()` to handle both embedding generation and Qdrant search seamlessly in one call.
### Why: Minimizes manual embedding logic in the application layer and maintains exact compatibility with the Phase 1 Indexing implementation.
### Alternatives Considered: Manually instantiating `TextEmbedding` and passing raw vectors to `client.search()`.
### Why Alternatives Were Rejected: Redundant code. The `qdrant-client` 1.8.0 API simplifies this via the `.query()` method.
### Impact: `DenseRetriever` is concise and directly interoperable with the `apple_10k` collection created in Phase 1.
### Phase: Phase 2 (Milestone 2)

## [2026-09-18 00:43]
### Decision: Hybrid Retrieval with Concurrent Execution and Reciprocal Rank Fusion (RRF).
### Context: Phase 2 Milestone 3 requires executing both BM25 and Dense retrieval strategies concurrently and fusing the results.
### Decision Made: 
- Implemented `HybridRetriever` in `src/retrieval/hybrid.py`.
- Used `concurrent.futures.ThreadPoolExecutor(max_workers=2)` for parallel execution of retrievers.
- Added `provenance` and `rrf_score` to the `RetrievalResult` schema to preserve debug traceability.
- Used an RRF formula of `1 / (k + rank)` with `k=60` and deduplicated by stable `chunk_id`.
### Why: 
- `ThreadPoolExecutor` is lightweight and doesn't introduce async complexity to an otherwise synchronous codebase.
- RRF is mathematically robust and standard in modern hybrid RAGs.
### Alternatives Considered: `asyncio` for parallel execution.
### Why Alternatives Were Rejected: The rest of the app is purely synchronous; introducing `asyncio` would unnecessarily infect the call chain and require async adaptations for Qdrant and Pickle endpoints.
### Impact: Produces the required Top-20 fused candidate pool for cross-encoder reranking.
### Phase: Phase 2 (Milestone 3)

## [2026-09-18 01:14]
### Decision: Use `Xenova/ms-marco-MiniLM-L-6-v2` as the local Cross-Encoder.
### Context: Phase 2 Milestone 4 requires implementing the Cross-Encoder reranker. The originally planned `BAAI/bge-reranker-base` is a 1.1GB ONNX model, which failed to reliably download via the HuggingFace CDN in this environment due to timeouts.
### Decision Made: 
- Upgraded `fastembed` to `0.8.0`.
- Used `fastembed.rerank.cross_encoder.TextCrossEncoder` with the fallback model `Xenova/ms-marco-MiniLM-L-6-v2`.
- Removed the silent fallback mechanism; if initialization fails, the pipeline now explicitly raises a `RuntimeError` rather than returning an unreranked list.
- Preserved the existing `RetrievalResult` schema but added a distinct `rerank_score`.
### Why: 
- `Xenova/ms-marco-MiniLM-L-6-v2` is only ~86MB and successfully initializes without HF timeout issues on this bandwidth constrained environment, while remaining fully compatible with the FastEmbed ONNX runtime.
- Explicitly separating the scores allows verification that real cross-encoder relevance scores were generated, without muddying the provenance of the RRF score.
### Tradeoffs: 
- `ms-marco-MiniLM-L-6-v2` has a lower theoretical ranking capacity compared to `bge-reranker-base`. However, operational reliability strictly takes precedence here to allow verification of the architectural data flow. The embedding model used for dense retrieval remains unchanged as `BAAI/bge-small-en-v1.5`.
### Impact: The retrieval pipeline correctly initializes and generates actual rerank scores for the fused top-20 pool, completing Phase 2 end-to-end verification.
### Phase: Phase 2 (Milestone 4)
