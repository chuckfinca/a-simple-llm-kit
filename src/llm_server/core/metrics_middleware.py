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
            "/v1/predict",
            "/v1/pipeline",
        ]
        logging.info(
            f"Simplified PerformanceMetricsMiddleware initialized with tracked paths: {self.tracked_paths}"
        )

    def _modify_response_body(self, body_bytes: bytes) -> bytes:
        """
        Parses the response body, moves performance metrics to the top level,
        and returns the modified body bytes. Returns original bytes on failure.
        """
        try:
            response_data = json.loads(body_bytes)
            if not isinstance(response_data, dict):
                return body_bytes

            # Check for and extract performance metrics from the nested location
            metadata = response_data.get("data", {}).get("metadata", {})
            performance_metrics = metadata.get("performance_metrics")

            if not performance_metrics:
                return body_bytes  # No metrics to move, do nothing

            # Ensure top-level metadata exists
            if "metadata" not in response_data:
                response_data["metadata"] = {}

            # Move the metrics
            response_data["metadata"]["performance"] = performance_metrics
            del response_data["data"]["metadata"]["performance_metrics"]

            # Clean up empty metadata dict if necessary
            if not response_data["data"]["metadata"]:
                del response_data["data"]["metadata"]

            logging.debug("Moved pipeline metrics to top-level performance key.")
            return json.dumps(response_data).encode()

        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(
                f"Failed to modify response for metrics: {e}", exc_info=True
            )
            return body_bytes  # Return original body if anything goes wrong

    async def __call__(self, scope: dict, receive: Callable, send: Callable):
        if scope["type"] != "http" or not any(
            scope.get("path", "").startswith(p) for p in self.tracked_paths
        ):
            await self.app(scope, receive, send)
            return

        response_body = bytearray()
        original_start_message = {}

        async def send_interceptor(message: dict):
            nonlocal response_body, original_start_message
            if message["type"] == "http.response.start":
                # Don't send yet, just store it
                original_start_message = message
                return

            if message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))
                # If this is the last chunk, process and send the full response
                if not message.get("more_body", False):
                    # Modify the completed body
                    modified_body = self._modify_response_body(bytes(response_body))

                    # Update content-length header
                    headers = original_start_message.get("headers", [])
                    headers = [
                        (k, v) for k, v in headers if k.lower() != b"content-length"
                    ]
                    headers.append(
                        (b"content-length", str(len(modified_body)).encode())
                    )
                    original_start_message["headers"] = headers

                    # Now send the start message and the (possibly modified) body
                    await send(original_start_message)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": modified_body,
                            "more_body": False,
                        }
                    )
                return  # Absorb the original message

            await send(message)

        await self.app(scope, receive, send_interceptor)


def add_metrics_middleware(app):
    """Add performance metrics middleware to FastAPI application"""
    logging.info("Registering simplified PerformanceMetricsMiddleware")
    app.add_middleware(PerformanceMetricsMiddleware)
    return app
