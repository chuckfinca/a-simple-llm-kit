from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from llm_server.core import logging
from llm_server.core.config import Settings, get_settings

# Header name for API key in requests
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)


def get_api_key(
    request: Request,
    api_key_header: str = Security(api_key_header),
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    # Log the full path for debugging
    path = request.url.path

    # Check if the API key is configured
    if not settings.llm_server_api_key:
        logging.error(f"Path: {path} - API key is not configured in environment")
        raise HTTPException(
            status_code=500, detail="API key authentication is not properly configured"
        )

    # Log key comparison details (safely)
    expected_key = settings.llm_server_api_key
    expected_len = len(expected_key)
    provided_len = len(api_key_header)

    # Compare the provided key with the configured key
    if api_key_header == expected_key:
        logging.info(f"Path: {path} - Valid API key provided")
        return api_key_header
    else:
        logging.warning(
            f"Path: {path} - Invalid API key provided. "
            f"Expected length: {expected_len}, Provided length: {provided_len}. "
            f"First chars don't match: {api_key_header[:2] != expected_key[:2]}"
        )
        raise HTTPException(status_code=403, detail="Invalid API key")
