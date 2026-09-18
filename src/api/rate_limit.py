from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from src.config import Config

def get_client_id(request: Request) -> str:
    """
    Identity function for rate limiting. 
    Uses the hashed API key from the request state if authenticated,
    otherwise falls back to IP address.
    """
    if hasattr(request.state, "client_id") and request.state.client_id:
        return request.state.client_id
    return get_remote_address(request)

limiter = Limiter(key_func=get_client_id)
