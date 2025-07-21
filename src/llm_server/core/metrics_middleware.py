import json
from typing import Callable, Optional

from llm_server.core import logging


class PerformanceMetricsMiddleware:
    """
    Simplified ASGI-compliant middleware that standardizes performance metrics location.
    This version removes the legacy metrics collection and relies entirely on the pipeline metrics.
    """

    def __init__(self, app, tracked_paths: Optional[list] = None):
        self.app = app
        self.tracked_paths = tracked_paths or [
            "/v1/extract-contact",
            "/v1/predict",
            "/v1/pipeline",
        ]
        logging.info(
            f"Simplified PerformanceMetricsMiddleware initialized with tracked paths: {self.tracked_paths}"
        )

    async def __call__(self, scope: dict, receive: Callable, send: Callable):
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

        # Capture the response to modify
        response_started = False
        response_status = 200
        response_headers = []
        response_body = bytearray()

        # Create a send interceptor that captures the response
        async def send_interceptor(message: dict):
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
                            # Check if performance_metrics exist in data.metadata
                            if (
                                "data" in response_data
                                and isinstance(response_data["data"], dict)
                                and "metadata" in response_data["data"]
                                and isinstance(response_data["data"]["metadata"], dict)
                                and "performance_metrics"
                                in response_data["data"]["metadata"]
                            ):
                                # Get the enhanced metrics
                                enhanced_metrics = response_data["data"]["metadata"][
                                    "performance_metrics"
                                ]

                                # Move to top-level metadata and remove from data.metadata
                                if "metadata" in response_data:
                                    response_data["metadata"]["performance"] = (
                                        enhanced_metrics
                                    )
                                    del response_data["data"]["metadata"][
                                        "performance_metrics"
                                    ]
                                    logging.debug(
                                        "Moved pipeline metrics to top-level performance key"
                                    )

                                # If data.metadata is now empty, consider removing it
                                if not response_data["data"]["metadata"]:
                                    del response_data["data"]["metadata"]

                            # Convert back to JSON
                            modified_body = json.dumps(response_data).encode()

                            # Update the Content-Length header
                            updated_headers = []
                            content_length_updated = False

                            for header_name, header_value in response_headers:
                                if header_name.lower() == b"content-length":
                                    updated_headers.append(
                                        (header_name, str(len(modified_body)).encode())
                                    )
                                    content_length_updated = True
                                else:
                                    updated_headers.append((header_name, header_value))

                            # Add Content-Length if it wasn't there
                            if not content_length_updated:
                                updated_headers.append(
                                    (
                                        b"content-length",
                                        str(len(modified_body)).encode(),
                                    )
                                )

                            # Send the modified response
                            await send(
                                {
                                    "type": "http.response.start",
                                    "status": response_status,
                                    "headers": updated_headers,
                                }
                            )

                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": modified_body,
                                    "more_body": False,
                                }
                            )

                            return

                    except Exception as e:
                        logging.warning(
                            f"Failed to modify response: {e}", exc_info=True
                        )

                    # If anything failed, send the original response
                    await send(
                        {
                            "type": "http.response.start",
                            "status": response_status,
                            "headers": response_headers,
                        }
                    )

                    await send(
                        {
                            "type": "http.response.body",
                            "body": bytes(response_body),
                            "more_body": False,
                        }
                    )
            else:
                # Pass through other message types
                await send(message)

        # Simple pass-through for receive
        async def receive_interceptor():
            return await receive()

        try:
            await self.app(scope, receive_interceptor, send_interceptor)
        except Exception as e:
            logging.error(
                f"Error in metrics middleware for {path}: {str(e)}", exc_info=True
            )
            raise


def add_metrics_middleware(app):
    """Add performance metrics middleware to FastAPI application"""
    logging.info("Registering simplified PerformanceMetricsMiddleware")
    app.add_middleware(PerformanceMetricsMiddleware)
    return app
