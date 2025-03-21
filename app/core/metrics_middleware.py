from typing import Dict, Any, Optional, List, Callable
import json
import time
import uuid
from app.core import logging

class PerformanceMetrics:
    """Comprehensive performance metrics tracker"""
    
    def __init__(self):
        """Initialize a new metrics tracker with start time"""
        self.start_time = time.time()
        self.trace_id = str(uuid.uuid4())
        self.checkpoints = {"request_start": self.start_time}
        self.token_usage = {"input": 0, "output": 0, "total": 0}
        self.model_id = None
        self.model_info = {}
        self.metadata = {}
        self._latest_checkpoint = self.start_time
    
    def mark_checkpoint(self, name: str) -> float:
        """
        Mark a timing checkpoint and return elapsed time since last checkpoint
        
        Args:
            name: Name of the checkpoint
            
        Returns:
            Seconds elapsed since last checkpoint
        """
        now = time.time()
        elapsed = now - self._latest_checkpoint
        self.checkpoints[name] = now
        duration_key = f"{name}_duration"
        self.metadata[duration_key] = elapsed
        self._latest_checkpoint = now
        return elapsed
    
    def record_preparation_complete(self) -> None:
        """Mark when prompt/input preparation is complete"""
        self.mark_checkpoint('preparation_complete')
    
    def record_model_start(self) -> None:
        """Mark when model starts processing"""
        self.mark_checkpoint('model_start')
    
    def record_model_complete(self) -> None:
        """Mark when model completes processing"""
        self.mark_checkpoint('model_complete')
    
    def record_response_ready(self) -> None:
        """Mark when response is ready to be sent"""
        self.mark_checkpoint('response_ready')
    
    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """
        Record token usage information
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.token_usage["input"] = input_tokens
        self.token_usage["output"] = output_tokens
        self.token_usage["total"] = input_tokens + output_tokens
        
        # Store in metadata too
        self.metadata["input_tokens"] = input_tokens
        self.metadata["output_tokens"] = output_tokens
        self.metadata["total_tokens"] = input_tokens + output_tokens
        
        # Calculate estimated cost if we have the model info
        if self.model_id:
            self._calculate_cost()
    
    def set_model_info(self, model_id: str, model_info: Optional[Dict[str, Any]] = None) -> None:
        """Set model information for cost calculation"""
        self.model_id = model_id
        if model_info:
            self.model_info = model_info
        
        # Store in metadata
        self.metadata["model_id"] = model_id
        
        # Recalculate cost if we have token usage
        if self.token_usage["total"] > 0:
            self._calculate_cost()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata"""
        self.metadata[key] = value
    
    def _calculate_cost(self) -> None:
        """Calculate the estimated cost based on token usage and model"""
        # Cost rates per 1K tokens (input, output) in USD as of March 2025
        pricing = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (0.5, 1.5),
            "claude-3.7-sonnet": (3.0, 15.0),
            "gemini-2.0-flash": (0.35, 1.05),
            # Fallback rates for unknown models
            "default": (0.0015, 0.002)
        }
        
        # Get rate for the model, default to fallback rate if not found
        input_rate, output_rate = pricing.get(self.model_id, pricing["default"])
        
        # Calculate cost
        input_cost = (self.token_usage["input"] / 1000) * input_rate
        output_cost = (self.token_usage["output"] / 1000) * output_rate
        total_cost = input_cost + output_cost
        
        # Store in metadata
        self.metadata["estimated_cost_usd"] = round(total_cost, 6)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metrics for API responses"""
        # Calculate total time
        total_time = time.time() - self.start_time
        
        # Create summary with timing information
        timing = {
            "total_ms": round(total_time * 1000, 2),
        }
        
        # Add detailed timing for each checkpoint
        for name, timestamp in self.checkpoints.items():
            if name != "request_start":
                timing[f"{name}_ms"] = round((timestamp - self.start_time) * 1000, 2)
        
        # Calculate stage durations if checkpoints exist
        if "preparation_complete" in self.checkpoints and "request_start" in self.checkpoints:
            timing["preparation_ms"] = round((self.checkpoints["preparation_complete"] - self.checkpoints["request_start"]) * 1000, 2)
        
        if "model_start" in self.checkpoints and "preparation_complete" in self.checkpoints:
            timing["queue_ms"] = round((self.checkpoints["model_start"] - self.checkpoints["preparation_complete"]) * 1000, 2)
        
        if "model_complete" in self.checkpoints and "model_start" in self.checkpoints:
            timing["model_execution_ms"] = round((self.checkpoints["model_complete"] - self.checkpoints["model_start"]) * 1000, 2)
        
        if "response_ready" in self.checkpoints and "model_complete" in self.checkpoints:
            timing["post_processing_ms"] = round((self.checkpoints["response_ready"] - self.checkpoints["model_complete"]) * 1000, 2)
        
        summary = {
            "timing": timing,
            "tokens": self.token_usage.copy(),
            "trace_id": self.trace_id,
        }
        
        # Add estimated cost if available
        if "estimated_cost_usd" in self.metadata:
            summary["estimated_cost_usd"] = self.metadata["estimated_cost_usd"]
        
        # Add model information if available
        if self.model_id:
            summary["model"] = {
                "id": self.model_id
            }
            if self.model_info:
                summary["model"].update(self.model_info)
        
        return summary
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics for logging"""
        all_metrics = {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "total_time": time.time() - self.start_time,
            "checkpoints": {k: v for k, v in self.checkpoints.items()},
            "token_usage": self.token_usage.copy(),
            "model_id": self.model_id,
        }
        
        # Add all metadata
        all_metrics.update(self.metadata)
        
        return all_metrics


class PerformanceMetricsMiddleware:
    """
    ASGI-compliant middleware that adds comprehensive performance metrics to API responses.
    """
    
    def __init__(
        self, 
        app,
        tracked_paths: list = None
    ):
        self.app = app
        self.tracked_paths = tracked_paths or ["/v1/extract-contact", "/v1/predict", "/v1/pipeline"]
        logging.info(f"PerformanceMetricsMiddleware initialized with tracked paths: {self.tracked_paths}")
    
    async def __call__(self, scope: Dict, receive: Callable, send: Callable):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Get the request path
        path = scope.get("path", "")
        
        # Check if this path should be tracked
        should_track = any(path.startswith(tracked) for tracked in self.tracked_paths)
        logging.debug(f"Request path: {path}, should track: {should_track}")
        
        if not should_track:
            await self.app(scope, receive, send)
            return
        
        logging.info(f"Starting performance tracking for path: {path}")
        
        # Create a metrics collector for this request
        metrics = PerformanceMetrics()
        
        # Add to request state
        if not hasattr(scope, "state"):
            scope["state"] = {}
        scope["state"]["metrics"] = metrics
        
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
                    # Mark response as ready
                    metrics.record_response_ready()
                    
                    try:
                        # Try to parse as JSON
                        response_data = json.loads(response_body)
                        
                        # Add metrics to the response
                        metrics_summary = metrics.get_summary()
                        logging.debug(f"Generated metrics summary: {json.dumps(metrics_summary)}")
                        
                        if isinstance(response_data, dict):
                            # Add to top-level metadata if it exists
                            if "metadata" in response_data:
                                logging.debug("Adding metrics to top-level metadata")
                                response_data["metadata"]["performance_metrics"] = metrics_summary
                            else:
                                logging.debug("No top-level metadata field found in response")
                            
                            # Add to nested data.metadata if it exists
                            if "data" in response_data and isinstance(response_data["data"], dict):
                                if "metadata" in response_data["data"]:
                                    logging.debug("Adding metrics to nested data.metadata")
                                    response_data["data"]["metadata"]["performance_metrics"] = metrics_summary
                                else:
                                    logging.debug("No nested data.metadata field found in response")
                            else:
                                logging.debug("No valid data field found in response")
                            
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
                            
                            logging.info(
                                f"Successfully added performance metrics to response for {path}",
                                extra={"performance_metrics": metrics.get_all_metrics()}
                            )
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
        
        # Create a receive interceptor to add metrics to request handling
        async def receive_interceptor():
            message = await receive()
            
            # If this is a request body, mark preparation start
            if message["type"] == "http.request":
                metrics.mark_checkpoint("request_body_received")
                
                # Try to extract model_id from request body for token estimation
                try:
                    if message.get("body"):
                        body = message.get("body", b"")
                        if body:
                            body_json = json.loads(body)
                            if isinstance(body_json, dict) and "request" in body_json:
                                req_data = body_json["request"]
                                if isinstance(req_data, dict):
                                    # Extract model_id from different possible locations
                                    model_id = None
                                    if "model_id" in req_data:
                                        model_id = req_data["model_id"]
                                    elif "params" in req_data and isinstance(req_data["params"], dict):
                                        model_id = req_data["params"].get("model_id")
                                    
                                    if model_id:
                                        metrics.set_model_info(model_id)
                except Exception as e:
                    # Just log but don't interfere with request processing
                    logging.debug(f"Failed to extract model_id from request: {str(e)}")
            
            return message
        
        try:
            await self.app(scope, receive_interceptor, send_interceptor)
            logging.debug(f"Completed processing request for {path}")
        except Exception as e:
            logging.error(f"Error in metrics middleware for {path}: {str(e)}", exc_info=True)
            raise


def add_metrics_middleware(app):
    """Add performance metrics middleware to FastAPI application"""
    logging.info("Registering PerformanceMetricsMiddleware")
    app.add_middleware(PerformanceMetricsMiddleware)
    return app