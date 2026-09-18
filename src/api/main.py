import uuid
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from src.api.logging_config import setup_structured_logging
from src.api.rate_limit import limiter
from src.api.routes import query, documents

# Set up logging early
setup_structured_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="HydraCache API", version="1.0.0")

# Rate limiter setup
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded"}
    )
    # Slowapi attaches the header on exc.headers if Retry-After is computed
    if hasattr(exc, "headers") and exc.headers:
        for k, v in exc.headers.items():
            response.headers[k] = v
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled Server Error", extra={"endpoint": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal Server Error"}
    )

@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    # Retrieve or generate Request ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    # Inject request_id into logging context via LogRecord Factory or directly on logger?
    # Simple approach for middleware: we just log it. A production app might use contextvars.
    # For now, we will log the start and end of request.
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "endpoint": request.url.path
        }
    )
    
    response = await call_next(request)
    
    response.headers["X-Request-ID"] = request_id
    
    logger.info(
        f"Request completed: {request.method} {request.url.path} - Status: {response.status_code}",
        extra={
            "request_id": request_id,
            "status_code": response.status_code
        }
    )
    
    return response

@app.on_event("startup")
async def startup_event():
    # Initialize the heavy pipeline once at startup
    from src.retrieval.answer import AnswerPipeline
    app.state.pipeline = AnswerPipeline()
    logger.info("HydraCache Pipeline Initialized")

# Include Routers
app.include_router(query.router, tags=["Query"])
app.include_router(documents.router, tags=["Documents"])

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Lightweight health check. 
    Checks if application is alive. Dependency checking (Qdrant/Redis) is skipped here 
    to avoid heavy blocking, but could be added as a separate /health/deep endpoint.
    """
    return {"status": "healthy"}
