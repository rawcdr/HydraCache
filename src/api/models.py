from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query.")

class Citation(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int
    chunk_type: str
    parent_id: Optional[str] = None
    retrieval_source: str
    provenance: Optional[List[str]] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None

class LatencyBreakdown(BaseModel):
    cache_latency: float
    retrieval_latency: float
    generation_latency: float
    total_latency: float

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class QueryResponse(BaseModel):
    answer: str
    cache_hit: bool
    cache_distance: Optional[float] = None
    query_mode: str = "single_hop"
    sub_questions: List[str] = Field(default_factory=list)
    latency: LatencyBreakdown
    tokens: TokenUsage
    context: Optional[List[Citation]] = None

class DocumentUploadResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str
