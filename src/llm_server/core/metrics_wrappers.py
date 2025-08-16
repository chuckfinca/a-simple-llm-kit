import time
import uuid
from typing import Any

from llm_server.core.context import _current_metrics

# [OTel] This block handles the optional import of OpenTelemetry components.
# If OTel is not installed or disabled via config, all instruments will be None.
from llm_server.core.opentelemetry_integration import (
    _OTEL_ENABLED,
    REQUEST_DURATION_SECONDS,
    _tracer,
    trace,
)
from llm_server.core.protocols import PipelineStep
from llm_server.core.types import MediaType, PipelineData, Usage


class PerformanceMetrics:
    """
    Tracks performance and observability data for a single request.
    This class acts as a container for in-flight observability data that is
    used to populate API response metadata and emit OpenTelemetry signals.
    """

    def __init__(self):
        """Initialize a new metrics tracker for a single request."""
        self.start_time = time.time()
        self.trace_id = str(uuid.uuid4())
        self.checkpoints = {"request_start": self.start_time}
        self.model_id: str | None = None
        self.model_info: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.usage: Usage | None = None
        self._latest_checkpoint = self.start_time
        _current_metrics.set(self)

        self._current_span = None
        if _OTEL_ENABLED and _tracer:
            self._current_span = _tracer.start_span("llm_request")
            self._current_span.set_attribute("trace_id", self.trace_id)

    def mark_checkpoint(self, name: str) -> float:
        """Mark a timing checkpoint."""
        now = time.time()
        elapsed = now - self._latest_checkpoint
        self.checkpoints[name] = now
        self._latest_checkpoint = now
        if self._current_span:
            self._current_span.add_event(
                f"checkpoint.{name}",
                {"duration_since_last_checkpoint_ms": elapsed * 1000},
            )
        return elapsed

    def set_model_info(
        self, model_id: str, model_info: dict[str, Any] | None = None
    ) -> None:
        """Set model info for the API response and OpenTelemetry span."""
        self.model_id = model_id
        if model_info:
            self.model_info = model_info
        if self._current_span:
            self._current_span.set_attribute("llm.request.model", model_id)
            if model_info and (provider := model_info.get("provider")):
                self._current_span.set_attribute("llm.vendor", provider)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to the API response."""
        self.metadata[key] = value

    def set_usage(self, usage: Usage) -> None:
        """
        Sets the token usage for this request's context. This populates
        the API response and emits OpenTelemetry attributes.
        """
        self.usage = usage
        if self._current_span:
            self._current_span.set_attributes(
                {
                    "llm.usage.input_tokens": usage.prompt_tokens,
                    "llm.usage.output_tokens": usage.completion_tokens,
                    "llm.usage.total_tokens": usage.prompt_tokens
                    + usage.completion_tokens,
                }
            )

    def finish(self, status: str = "success", program_id: str = "unknown"):
        """Finalizes tracking at the end of a request."""
        total_time = time.time() - self.start_time
        if _OTEL_ENABLED and REQUEST_DURATION_SECONDS:
            attributes = {
                "model_id": self.model_id or "unknown",
                "program_id": program_id,
                "status": status,
            }
            REQUEST_DURATION_SECONDS.record(total_time, attributes)
        if self._current_span:
            self._current_span.set_attribute("total_duration_s", total_time)
            self._current_span.set_attribute("status", status)
            self._current_span.end()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the metrics for the API response."""
        total_time = time.time() - self.start_time
        summary = {
            "timing": {"total_ms": round(total_time * 1000, 2)},
            "trace_id": self.trace_id,
        }
        if self.usage:
            summary["tokens"] = {
                "input": self.usage.prompt_tokens,
                "output": self.usage.completion_tokens,
                "total": self.usage.prompt_tokens + self.usage.completion_tokens,
            }
            if "estimated_cost_usd" in self.metadata:
                summary["tokens"]["cost_usd"] = self.metadata["estimated_cost_usd"]

        for name, timestamp in self.checkpoints.items():
            if name != "request_start":
                summary["timing"][f"{name}_ms"] = round(
                    (timestamp - self.start_time) * 1000, 2
                )
        return summary


class PipelineStepTracker:
    """Wrapper that adds OTel tracing to any PipelineStep implementation."""

    def __init__(self, step: PipelineStep, step_name: str | None = None):
        self.step = step
        self.step_name = step_name or step.__class__.__name__

    @property
    def accepted_media_types(self) -> list[MediaType]:
        return self.step.accepted_media_types

    async def process(self, data: PipelineData) -> PipelineData:
        """Process data within a dedicated OTel child span for the step."""
        if _OTEL_ENABLED and _tracer:
            with _tracer.start_as_current_span(
                f"pipeline.step.{self.step_name}"
            ) as span:
                span.set_attributes(
                    {
                        "pipeline.step.name": self.step_name,
                        "pipeline.step.input_type": data.media_type.value,
                    }
                )
                try:
                    result = await self.step.process(data)
                    span.set_attribute(
                        "pipeline.step.output_type", result.media_type.value
                    )
                    span.set_status(trace.StatusCode.OK)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
        else:
            return await self.step.process(data)
