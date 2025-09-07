import time
import uuid
from typing import Any

from a_simple_llm_kit.core.context import _current_metrics
from a_simple_llm_kit.core.opentelemetry_integration import (
    _OTEL_ENABLED,
    REQUEST_DURATION_SECONDS,
    _tracer,
    trace,
)
from a_simple_llm_kit.core.protocols import PipelineStep
from a_simple_llm_kit.core.types import (
    MediaType,
    ModelInfo,
    PerformanceSummary,
    PipelineData,
    TokenSummary,
    Usage,
)


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
        self, model_id: str, model_info: ModelInfo | None = None
    ) -> None:
        """Set model info for the API response and OpenTelemetry span."""
        self.model_id = model_id
        if model_info:
            if self._current_span:
                self._current_span.set_attribute("llm.request.model", model_id)
                self._current_span.set_attribute("llm.vendor", model_info.provider)
        elif self._current_span:
            self._current_span.set_attribute("llm.request.model", model_id)

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
        """Get a summary of the metrics for the API response using Pydantic models."""
        total_time = time.time() - self.start_time

        timing_data = {"total_ms": round(total_time * 1000, 2)}
        for name, timestamp in self.checkpoints.items():
            if name != "request_start":
                timing_data[f"{name}_ms"] = round(
                    (timestamp - self.start_time) * 1000, 2
                )

        token_data = None
        if self.usage:
            token_data = {
                "input_tokens": self.usage.prompt_tokens,
                "output_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.prompt_tokens + self.usage.completion_tokens,
            }
            if "estimated_cost_usd" in self.metadata:
                token_data["cost_usd"] = self.metadata["estimated_cost_usd"]

        tokens_summary = (
            TokenSummary(
                prompt_tokens=self.usage.prompt_tokens,
                completion_tokens=self.usage.completion_tokens,
                total_tokens=self.usage.prompt_tokens + self.usage.completion_tokens,
                cost_usd=self.metadata.get("estimated_cost_usd"),
            )
            if self.usage
            else None
        )

        summary_model = PerformanceSummary(
            trace_id=self.trace_id, timing=timing_data, tokens=tokens_summary
        )

        return summary_model.model_dump(by_alias=True, exclude_none=True)


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
