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
