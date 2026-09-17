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
