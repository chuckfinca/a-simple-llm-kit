from fastapi import Request, Response
from fastapi.responses import JSONResponse
from typing import Callable, Dict, Any, List
import json
from app.core import logging

class VersioningMiddleware:
    """
    Middleware that ensures all API responses include program and model versioning information.
    Updated to support a consistent metadata structure.
    """
    
    def __init__(
        self, 
        app,
        required_fields: Dict[str, str] = None,
        bypass_paths: List[str] = None
    ):
        self.app = app
        self.bypass_paths = bypass_paths or ["/v1/health", "/docs", "/openapi.json", "/redoc"]
        self.required_fields = required_fields or {
            "program_id": "Program ID must be included in response metadata",
            "program_version": "Program version must be included in response metadata"
        }
    
    async def __call__(self, scope: Dict, receive: Callable, send: Callable):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Get the request path
        path = scope.get("path", "")
        
        # Check if this path should be bypassed
        if any(path.startswith(bypass) for bypass in self.bypass_paths):
            await self.app(scope, receive, send)
            return
        
        # Create a response interceptor
        async def _send(message: Dict):
            if message["type"] == "http.response.start":
                # Store the status code
                self.status_code = message.get("status", 200)
                
            elif message["type"] == "http.response.body" and self.status_code < 300:
                # Only check successful responses (status < 300)
                body = message.get("body", b"{}")
                
                try:
                    # Attempt to parse response as JSON
                    response_data = json.loads(body)
                    
                    # Check if metadata exists
                    metadata = response_data.get("metadata", {})
                    
                    # Check for required fields
                    missing_fields = []
                    for field, error_message in self.required_fields.items():
                        # Check for standardized nested structure (new approach)
                        if field == "program_id" and "program" in metadata and "id" in metadata["program"]:
                            continue
                        elif field == "program_version" and "program" in metadata and "version" in metadata["program"]:
                            continue
                        elif field == "model_id" and "model" in metadata and "id" in metadata["model"]:
                            continue
                        # Legacy flat structure support (transitional)
                        elif field in metadata:
                            # Found in flat structure, but log a warning to encourage migration
                            logging.warning(
                                f"Legacy metadata format detected in {path}. "
                                f"Field '{field}' should be in nested structure."
                            )
                            continue
                            
                        # Field not found in either structure
                        missing_fields.append(error_message)
                    
                    if missing_fields:
                        # Create error response
                        error_message = "; ".join(missing_fields)
                        logging.error(f"Versioning error in {path}: {error_message}")
                        
                        error_response = {
                            "success": False,
                            "error": "Missing versioning information",
                            "details": error_message,
                            "status_code": 500
                        }
                        
                        # Replace the response with error response
                        message["body"] = json.dumps(error_response).encode()
                except Exception as e:
                    # If we can't parse the response as JSON, let it pass through
                    logging.warning(f"Could not validate versioning info in response: {str(e)}")
            
            await send(message)
        
        # Process the request with the intercepted send function
        await self.app(scope, receive, _send)


def add_versioning_middleware(app):
    """
    Add the versioning middleware to the FastAPI application.
    """
    app.add_middleware(
        VersioningMiddleware,
        # Updated field names to match the new nested structure
        required_fields={
            "program_id": "Program ID must be included in response metadata",
            "program_version": "Program version must be included in response metadata",
            "model_id": "Model ID must be included in response metadata"
        },
        bypass_paths=[
            "/v1/health", 
            "/docs", 
            "/openapi.json", 
            "/redoc",
            "/v1/programs",
            "/metrics",     
            "/debug/orientation" 
        ]
    )