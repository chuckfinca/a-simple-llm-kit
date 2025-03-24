from fastapi import Request, Response
from typing import Dict, Any, Optional, List, Callable
import json
import time
from app.core import logging

class LatencyTrackingMiddleware:
    """
    ASGI-compliant middleware that tracks request latency including cold start time.
    """
    
    def __init__(
        self, 
        app,
        timestamp_header: str = "X-Request-Time",
        tracked_paths: list = None
    ):
        self.app = app
        self.timestamp_header = timestamp_header
        self.tracked_paths = tracked_paths or ["/v1/extract-contact", "/v1/predict", "/v1/pipeline"]
        logging.info(f"LatencyTrackingMiddleware initialized with tracked paths: {self.tracked_paths}")
    
    async def __call__(self, scope: Dict, receive: Callable, send: Callable):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Get the request path
        path = scope.get("path", "")
        
        # Check if this path should be tracked
        should_track = any(path.startswith(tracked) for tracked in self.tracked_paths)
        
        if not should_track:
            await self.app(scope, receive, send)
            return
        
        # Get server-side start time
        server_start_time = time.time() * 1000  # milliseconds
        
        # The client timestamp will be available in request.state after the dependency runs
        # For now, set initial values
        client_timestamp = None
        initial_latency = None
        
        # We'll capture the timestamp from request state in a later step
        # This is just a fallback mechanism in case the dependency somehow gets bypassed
        for header_name, header_value in scope.get("headers", []):
            if header_name.decode("utf-8").lower() == self.timestamp_header.lower():
                try:
                    client_timestamp = float(header_value.decode("utf-8"))
                except (ValueError, TypeError):
                    logging.warning(f"Invalid {self.timestamp_header} header value: {header_value}")
        
        # Request object will be available after the first receive
        async def get_request_state():
            nonlocal client_timestamp, initial_latency
            
            # Get access to the request object to check state
            from starlette.requests import Request
            request = Request(scope, receive, send)
            
            # Try to get validated timestamp from request state (set by the dependency)
            if hasattr(request.state, "client_timestamp"):
                client_timestamp = request.state.client_timestamp
                initial_latency = server_start_time - client_timestamp
                logging.debug(f"Using validated timestamp from dependency: {client_timestamp}")
            elif client_timestamp:
                # Fallback to header value if dependency somehow didn't run
                initial_latency = server_start_time - client_timestamp
                logging.debug(f"Using timestamp from header: {client_timestamp}")
            else:
                logging.warning(f"No client timestamp available for {path}")
        
        # Capture the response to modify
        response_started = False
        response_status = 200
        response_headers = []
        response_body = bytearray()
        
        # Create a send interceptor that captures the response
        async def send_interceptor(message: Dict):
            nonlocal response_started, response_status, response_headers, response_body
            
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message.get("status", 200)
                response_headers = message.get("headers", [])
                
                # For non-success responses, just pass through
                if response_status >= 300:
                    await send(message)
                    
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                more_body = message.get("more_body", False)
                
                # If this is a streaming response or non-success, just pass through
                if more_body or response_status >= 300:
                    await send(message)
                    return
                
                # Append to the complete body
                response_body.extend(body)
                
                # If this is the final body chunk, we can modify and send the complete response
                if not more_body:
                    try:
                        # Try to parse as JSON
                        response_data = json.loads(response_body)
                        
                        if isinstance(response_data, dict):
                            # Add latency metrics to response metadata
                            if "metadata" not in response_data:
                                response_data["metadata"] = {}
                                
                            if "performance" not in response_data["metadata"]:
                                response_data["metadata"]["performance"] = {}
                                
                            # Make sure timing object exists
                            if "timing" not in response_data["metadata"]["performance"]:
                                response_data["metadata"]["performance"]["timing"] = {}
                                
                            # Add initial latency if available
                            if initial_latency is not None:
                                latency_ms = round(initial_latency, 2)
                                response_data["metadata"]["performance"]["timing"]["initial_latency_ms"] = latency_ms
                                
                                # Update total_ms to include cold start/initial latency
                                if "total_ms" in response_data["metadata"]["performance"]["timing"]:
                                    # Store original processing time in case it's useful
                                    processing_ms = response_data["metadata"]["performance"]["timing"]["total_ms"]
                                    response_data["metadata"]["performance"]["timing"]["processing_ms"] = processing_ms
                                    
                                    # Update total_ms to be truly end-to-end
                                    response_data["metadata"]["performance"]["timing"]["total_ms"] = round(latency_ms + processing_ms, 2)
                                
                                logging.debug(f"Added initial latency metric: {latency_ms}ms")
                            
                            # Convert back to JSON
                            modified_body = json.dumps(response_data).encode()
                            
                            # Update the Content-Length header
                            updated_headers = []
                            content_length_updated = False
                            
                            for header_name, header_value in response_headers:
                                if header_name.lower() == b"content-length":
                                    updated_headers.append((header_name, str(len(modified_body)).encode()))
                                    content_length_updated = True
                                else:
                                    updated_headers.append((header_name, header_value))
                            
                            # Add Content-Length if it wasn't there
                            if not content_length_updated:
                                updated_headers.append((b"content-length", str(len(modified_body)).encode()))
                            
                            # Send the modified response
                            await send({
                                "type": "http.response.start",
                                "status": response_status,
                                "headers": updated_headers
                            })
                            
                            await send({
                                "type": "http.response.body",
                                "body": modified_body,
                                "more_body": False
                            })
                            
                            return
                        
                    except Exception as e:
                        logging.warning(f"Failed to modify response: {e}", exc_info=True)
                    
                    # If anything failed, send the original response
                    await send({
                        "type": "http.response.start",
                        "status": response_status,
                        "headers": response_headers
                    })
                    
                    await send({
                        "type": "http.response.body",
                        "body": bytes(response_body),
                        "more_body": False
                    })
            else:
                # Pass through other message types
                await send(message)
        
        # Intercept first receive to extract request state
        first_receive_done = False
        
        async def receive_interceptor():
            nonlocal first_receive_done
            
            message = await receive()
            
            # Only run once, after the first receive which typically has headers
            if not first_receive_done and message["type"] == "http.request":
                first_receive_done = True
                # Try to get timestamp from request state
                await get_request_state()
            
            return message
        
        try:
            await self.app(scope, receive_interceptor, send_interceptor)
        except Exception as e:
            logging.error(f"Error in latency tracking middleware for {path}: {str(e)}", exc_info=True)
            raise

def add_latency_tracking_middleware(app):
    """Add latency tracking middleware to FastAPI application"""
    logging.info("Registering LatencyTrackingMiddleware")
    app.add_middleware(LatencyTrackingMiddleware)
    return app