from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from app.core.config import Settings, get_settings

# Header name for API key in requests
API_KEY_NAME = "X-API-Key"  # This is what clients will use in HTTP headers
api_key_header = APIKeyHeader(name=API_KEY_NAME)

def get_api_key(
    api_key_header: str = Security(api_key_header),
    settings: Settings = Depends(get_settings)
) -> str:
    if api_key_header == settings.llm_server_api_key:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Invalid API key"
    )