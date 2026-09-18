import os
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict
from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException, BackgroundTasks, status
from src.api.models import DocumentUploadResponse, JobStatusResponse
from src.api.auth import get_api_key
from src.api.rate_limit import limiter
from src.config import Config

# For Phase 5, we use in-memory state tracking for simplicity as per specs.
# In a distributed production env, this would be Redis/Postgres + Celery.
job_registry: Dict[str, dict] = {}

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024 # 50 MB

def run_ingestion_pipeline(job_id: str, file_path: str):
    """Background task that executes the Phase 1 pipeline."""
    job_registry[job_id]["status"] = "running"
    job_registry[job_id]["started_at"] = datetime.utcnow().isoformat() + "Z"
    
    try:
        logger.info(f"Starting ingestion job {job_id} for file {file_path}")
        
        # We invoke the existing ingestion pipeline components
        from src.ingestion.parser import PDFParser
        from src.ingestion.chunker import TextChunker
        from src.indexing.dense import DenseIndexer
        from src.indexing.sparse import SparseIndexer
        
        parser = PDFParser(file_path)
        docs = parser.parse()
        
        chunker = TextChunker()
        chunks = chunker.chunk_documents(docs)
        
        dense_indexer = DenseIndexer()
        dense_indexer.index(chunks)
        
        sparse_indexer = SparseIndexer()
        sparse_indexer.index(chunks)
        
        job_registry[job_id]["status"] = "completed"
        logger.info(f"Ingestion job {job_id} completed successfully.")
        
    except Exception as e:
        logger.error(f"Ingestion job {job_id} failed: {e}")
        job_registry[job_id]["status"] = "failed"
        job_registry[job_id]["error"] = str(e)
        
    finally:
        job_registry[job_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"

@router.post("/documents", response_model=DocumentUploadResponse)
@limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    client_id: str = Depends(get_api_key)
):
    request.state.client_id = client_id
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
        
    # Check file size without reading entirely into memory if possible, 
    # but for simplicity we read and check
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max size is 50MB.")
        
    job_id = str(uuid.uuid4())
    
    # Save the file temporarily
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{job_id}.pdf"
    
    with open(file_path, "wb") as f:
        f.write(content)
        
    # Register job
    job_registry[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
        "error": None
    }
    
    # Enqueue background task
    background_tasks.add_task(run_ingestion_pipeline, job_id, file_path)
    
    logger.info(f"Document uploaded and job queued", extra={"job_id": job_id})
    return DocumentUploadResponse(job_id=job_id, status="queued", message="Document ingestion started.")

@router.get("/documents/{job_id}", response_model=JobStatusResponse)
@limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
async def get_job_status(
    request: Request,
    job_id: str,
    client_id: str = Depends(get_api_key)
):
    request.state.client_id = client_id
    
    if job_id not in job_registry:
        raise HTTPException(status_code=404, detail="Job ID not found.")
        
    job = job_registry[job_id]
    return JobStatusResponse(**job)
