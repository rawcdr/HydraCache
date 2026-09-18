import logging
import hashlib
from fastapi import APIRouter, Depends, Request
from src.api.models import QueryRequest, QueryResponse, Citation, LatencyBreakdown, TokenUsage
from src.api.auth import get_api_key
from src.api.rate_limit import limiter
from src.config import Config

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
@limiter.limit(f"{Config.RATE_LIMIT_PER_MINUTE}/minute")
async def query_endpoint(
    request: Request,
    payload: QueryRequest,
    client_id: str = Depends(get_api_key)
):
    # Store client_id in state for the rate limiter to use
    request.state.client_id = client_id
    
    # Safe logging identifier for the query (hash)
    query_hash = hashlib.sha256(payload.query.encode()).hexdigest()[:8]
    
    logger.info(
        f"Processing query.", 
        extra={
            "query_hash": query_hash,
            "endpoint": "/query",
            "method": "POST"
        }
    )
    
    pipeline = request.app.state.pipeline
    
    # Execute actual RAG pipeline
    result = pipeline.answer(payload.query)
    
    logger.info(
        f"Query completed.",
        extra={
            "query_hash": query_hash,
            "cache_hit": result.cache_hit,
            "query_mode": result.query_mode,
            "num_sub_questions": len(result.sub_questions) if result.sub_questions else 0,
            "latency": result.total_latency
        }
    )
    
    # Map to Pydantic Response
    citations = []
    if result.context:
        for c in result.context:
            citations.append(Citation(
                chunk_id=c.chunk_id,
                text=c.text,
                score=c.score,
                source_file=c.source_file,
                page_number=c.page_number,
                chunk_type=c.chunk_type,
                parent_id=c.parent_id,
                retrieval_source=c.retrieval_source,
                provenance=c.provenance,
                rrf_score=c.rrf_score,
                rerank_score=c.rerank_score
            ))
            
    return QueryResponse(
        answer=result.answer,
        cache_hit=result.cache_hit,
        cache_distance=result.cache_distance,
        query_mode=result.query_mode,
        sub_questions=result.sub_questions or [],
        latency=LatencyBreakdown(
            cache_latency=result.cache_latency,
            retrieval_latency=result.retrieval_latency,
            generation_latency=result.generation_latency,
            total_latency=result.total_latency
        ),
        tokens=TokenUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens
        ),
        context=citations
    )
