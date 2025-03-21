from fastapi import Request, Response
from typing import Callable, Dict, Any
import json
import time
from app.core import logging
from app.core.metrics_wrappers import PerformanceMetrics

class PerformanceMetricsMiddleware:
    """
    Middleware that adds performance metrics tracking to API endpoints.
    """
    
    def __init__(
        self, 
        app,
        tracked_paths: list = None
    ):
        self.app = app
        self.tracked_paths = tracked_paths or ["/v1/extract-contact", "/v1/predict", "/v1/pipeline"]
    
    async def __call__(self, scope: Dict, receive: Callable, send: Callable):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Get the request path
        path = scope.get("path", "")
        
        # Check if this path should be tracked
        if not any(path.startswith(tracked) for tracked in self.tracked_paths):
            await self.app(scope, receive, send)
            return
        
        # Create a metrics collector for this request
        metrics = PerformanceMetrics()
        
        # Add to request state
        if not hasattr(scope, "state"):
            scope["state"] = {}
        scope["state"]["metrics"] = metrics
        
        # Track response-related state between calls to _send
        response_state = {
            "status_code": 200,  # Default status code
            "headers": [],
            "content_length_index": -1,
            "seen_start": False
        }
        
        # Create a response interceptor
        async def _send(message: Dict):
            if message["type"] == "http.response.start":
                # Store the status code in our shared state
                response_state["status_code"] = message.get("status", 200)
                response_state["seen_start"] = True
                
                # Keep track of the headers
                response_state["headers"] = message.get("headers", [])
                
                # Find Content-Length header if present
                for i, (name, value) in enumerate(response_state["headers"]):
                    if name.decode("latin-1").lower() == "content-length":
                        response_state["content_length_index"] = i
                        break
                
                # Pass through without modification
                await send(message)
                
            elif message["type"] == "http.response.body":
                body = message.get("body", b"{}")
                more_body = message.get("more_body", False)
                
                # Only modify if not streaming and is a successful response
                if (not more_body and 
                    response_state["seen_start"] and 
                    response_state["status_code"] < 300):
                    try:
                        # Mark response as ready
                        metrics.mark_checkpoint("response_ready")
                        
                        # Attempt to parse response as JSON
                        response_data = json.loads(body)
                        
                        # Add metrics to response metadata
                        if isinstance(response_data, dict) and "metadata" in response_data:
                            # Get metrics
                            metrics_summary = metrics.get_summary()
                            
                            # Add to response metadata
                            response_data["metadata"]["performance_metrics"] = metrics_summary
                            
                            # Create updated body
                            new_body = json.dumps(response_data).encode()
                            
                            # Update Content-Length header if it exists
                            if response_state["content_length_index"] >= 0:
                                response_state["headers"][response_state["content_length_index"]] = (
                                    b"content-length", 
                                    str(len(new_body)).encode()
                                )
                                
                                # Need to send updated headers in a new start message
                                await send({
                                    "type": "http.response.start",
                                    "status": response_state["status_code"],
                                    "headers": response_state["headers"]
                                })
                            
                            # Update body with new content
                            message["body"] = new_body
                            
                            # Log for troubleshooting
                            logging.info(
                                f"Performance metrics for {path}",
                                extra={"performance_metrics": metrics.get_all_metrics()}
                            )
                    except Exception as e:
                        # Log but don't modify the response
                        logging.warning(f"Could not add metrics to response: {str(e)}")
                
                # Send the (possibly modified) body message
                await send(message)
            else:
                # Pass through other message types unchanged
                await send(message)
        
        # Process the request
        await self.app(scope, receive, _send)


def add_metrics_middleware(app):
    """Add performance metrics middleware to FastAPI application"""
    app.add_middleware(PerformanceMetricsMiddleware)