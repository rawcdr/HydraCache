import hashlib
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.config import Config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """
    Validate the API key from the header.
    Returns a hashed client_id for safe logging and rate limiting.
    """
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )
    
    if api_key_header != Config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
        
    # Return a safe, non-reversible hash of the key as the client identifier
    return hashlib.sha256(api_key_header.encode()).hexdigest()[:12]
