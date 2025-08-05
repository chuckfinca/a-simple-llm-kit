import time
import uuid
from typing import Any

from llm_server.core import logging
from llm_server.core.context import _current_metrics

# [OTel] This block handles the optional import of OpenTelemetry components.
# If OTel is not installed or disabled via config, all instruments will be None.
from llm_server.core.opentelemetry_integration import (
    _OTEL_ENABLED,
    MODEL_CALLS_TOTAL,
    REQUEST_DURATION_SECONDS,
    SpanKind,
    _tracer,
    trace,
)
from llm_server.core.protocols import ModelBackend, PipelineStep
from llm_server.core.types import MediaType, PipelineData


class PerformanceMetrics:
    """
    Tracks detailed performance metrics for a single request.
    This class is the central hub for observability, creating a detailed JSON
    summary for API responses AND emitting OpenTelemetry signals.
    """

    def __init__(self):
        """Initialize a new metrics tracker for a single request."""
        self.start_time = time.time()
        self.trace_id = str(uuid.uuid4())
        self.checkpoints = {"request_start": self.start_time}
        self.token_usage = {"input": 0, "output": 0, "total": 0}
        self.model_id: str | None = None
        self.model_info: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self._latest_checkpoint = self.start_time

        # [OTel] OTel Span Management and Context Propagation
        _current_metrics.set(self)

        self._current_span = None
        if _OTEL_ENABLED and _tracer:
            # Start the parent span for the entire request lifecycle.
            self._current_span = _tracer.start_span("llm_request")
            self._current_span.set_attribute("trace_id", self.trace_id)

    def mark_checkpoint(self, name: str) -> float:
        """Mark a timing checkpoint for the JSON summary and add an OTel span event."""
        now = time.time()
        elapsed = now - self._latest_checkpoint
        self.checkpoints[name] = now
        duration_key = f"{name}_duration"
        self.metadata[duration_key] = elapsed
        self._latest_checkpoint = now

        # [OTel] Add a point-in-time event to the parent span's timeline.
        if self._current_span:
            self._current_span.add_event(
                f"checkpoint.{name}",
                {"duration_since_last_checkpoint_ms": elapsed * 1000},
            )

        return elapsed

    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for the JSON summary and add attributes to the OTel span."""
        self.token_usage["input"] = input_tokens
        self.token_usage["output"] = output_tokens
        self.token_usage["total"] = input_tokens + output_tokens

        self.metadata["input_tokens"] = input_tokens
        self.metadata["output_tokens"] = output_tokens
        self.metadata["total_tokens"] = input_tokens + output_tokens

        # [OTel] Add attributes using the official OTel LLM semantic conventions.
        if self._current_span:
            self._current_span.set_attributes(
                {
                    "llm.usage.input_tokens": input_tokens,
                    "llm.usage.output_tokens": output_tokens,
                    "llm.usage.total_tokens": input_tokens + output_tokens,
                }
            )

        if self.model_id:
            self._calculate_cost()

    def set_model_info(
        self, model_id: str, model_info: dict[str, Any] | None = None
    ) -> None:
        """Set model info for the JSON summary and add attributes to the OTel span."""
        self.model_id = model_id
        if model_info:
            self.model_info = model_info

        self.metadata["model_id"] = model_id

        # [OTel] Add model info as semantic attributes to the parent span.
        if self._current_span:
            self._current_span.set_attribute("llm.request.model", model_id)
            if model_info:
                provider = model_info.get("provider")
                if provider:
                    self._current_span.set_attribute("llm.vendor", provider)

        if self.token_usage["total"] > 0:
            self._calculate_cost()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to the JSON summary."""
        self.metadata[key] = value

    def _calculate_cost(self) -> None:
        """Calculate estimated cost for the JSON summary and add to the OTel span."""
        pricing = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.5, 10.0),
            "claude-3.7-sonnet": (3.0, 15.0),
            "default": (1.5, 2.0),
        }
        key = self.model_id if self.model_id in pricing else "default"
        input_rate, output_rate = pricing[key]
        input_cost = (self.token_usage["input"] / 1_000_000) * input_rate
        output_cost = (self.token_usage["output"] / 1_000_000) * output_rate
        total_cost = input_cost + output_cost
        self.metadata["estimated_cost_usd"] = round(total_cost, 6)

        # [OTel] Add cost as a custom attribute to the span.
        if self._current_span:
            self._current_span.set_attribute(
                "llm.cost_usd", self.metadata["estimated_cost_usd"]
            )

    def finish(self, status: str = "success", program_id: str = "unknown"):
        """
        Finalizes tracking: ends the OTel span and records the final duration metric.
        This should be called by the application at the very end of a request.
        """
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
        """Get a summary of the metrics for API responses. This logic is preserved."""
        total_time = time.time() - self.start_time
        tokens_data = self.token_usage.copy()
        if "token_count_method" in self.metadata:
            tokens_data["method"] = self.metadata["token_count_method"]
        if "estimated_cost_usd" in self.metadata:
            tokens_data["cost_usd"] = self.metadata["estimated_cost_usd"]
        summary = {
            "timing": {"total_ms": round(total_time * 1000, 2)},
            "tokens": tokens_data,
            "trace_id": self.trace_id,
        }
        for name, timestamp in self.checkpoints.items():
            if name != "request_start":
                summary["timing"][f"{name}_ms"] = round(
                    (timestamp - self.start_time) * 1000, 2
                )
        if "step_timing" in self.metadata:
            summary["step_timing"] = self.metadata["step_timing"]
        return summary

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics for detailed logging. This logic is preserved."""
        all_metrics = {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "total_time": time.time() - self.start_time,
            "checkpoints": self.checkpoints.copy(),
            "token_usage": self.token_usage.copy(),
            "model_id": self.model_id,
        }
        all_metrics.update(self.metadata)
        return all_metrics


class MetricsTrackingLM:
    """
    DSPy LM wrapper that tracks metrics while preserving the full DSPy LM interface.
    
    This class acts as a transparent proxy to a DSPy LM, intercepting calls to track
    performance metrics without interfering with DSPy's internal mechanisms.
    """
    
    def __init__(self, original_lm, metrics: PerformanceMetrics):
        """
        Initialize the metrics-tracking LM wrapper.
        
        Args:
            original_lm: The original DSPy LM instance
            metrics: PerformanceMetrics instance for tracking
        """
        self.original_lm = original_lm
        self.metrics = metrics
        
        # Forward all non-callable attributes to the original LM
        # This ensures compatibility with DSPy's expectations
        for attr in dir(original_lm):
            if not attr.startswith('_') and not callable(getattr(original_lm, attr)):
                setattr(self, attr, getattr(original_lm, attr))
    
    def __call__(self, *args, **kwargs):
        """
        Track DSPy LM calls and extract metrics.
        
        This method intercepts the main LM call to track timing and token usage.
        """
        self.metrics.mark_checkpoint("model_start")
        try:
            result = self.original_lm(*args, **kwargs)
            self.metrics.mark_checkpoint("model_complete")
            
            # Extract token usage from DSPy history
            self._extract_and_record_tokens()
            
            return result
        except Exception as e:
            self.metrics.mark_checkpoint("model_error")
            raise
    
    def _extract_and_record_tokens(self):
        """
        Extract token usage from DSPy LM history and record in metrics.
        
        This method attempts multiple strategies to get accurate token counts,
        falling back to estimation if exact counts are unavailable.
        """
        try:
            if hasattr(self.original_lm, 'history') and self.original_lm.history:
                last_call = self.original_lm.history[-1]
                usage = last_call.get("usage", {})
                prompt_tokens = last_call.get("prompt_tokens") or usage.get("prompt_tokens")
                completion_tokens = last_call.get("completion_tokens") or usage.get("completion_tokens")
                
                if prompt_tokens is not None and completion_tokens is not None:
                    self.metrics.record_token_usage(prompt_tokens, completion_tokens)
                    self.metrics.add_metadata("token_count_method", "dspy_history_exact")
                    return
        except Exception as e:
            logging.warning(f"Could not extract exact token usage from DSPy history: {e}")
        
        # Fallback to estimation if exact tokens unavailable
        logging.info("Using fallback token estimation method")
        self.metrics.add_metadata("token_count_method", "estimation_fallback")
    
    def __getattr__(self, name):
        """
        Forward any other attribute access to the original LM.
        
        This ensures full compatibility with DSPy's LM interface by transparently
        forwarding all method calls and property accesses.
        """
        return getattr(self.original_lm, name)


class ModelBackendTracker:
    """Wrapper that adds OTel tracing and metrics to any ModelBackend implementation."""

    def __init__(self, backend: ModelBackend, metrics: PerformanceMetrics):
        self.backend = backend
        self.metrics = metrics
        self.model_id = getattr(backend, "model_id", "unknown")
        self.metrics.set_model_info(self.model_id, getattr(backend, "model_info", {}))
        if hasattr(backend, "program_metadata"):
            self.program_metadata = backend.program_metadata

    async def predict(self, input: Any) -> Any:
        """Execute prediction within a dedicated OTel child span."""
        if _OTEL_ENABLED and _tracer and SpanKind and trace:
            with _tracer.start_as_current_span(
                "llm.generate", kind=SpanKind.CLIENT
            ) as span:
                span.set_attributes(
                    {
                        "llm.vendor": self.metrics.model_info.get(
                            "provider", "unknown"
                        ),
                        "llm.request.model": self.model_id,
                    }
                )
                try:
                    self.metrics.mark_checkpoint("model_start")
                    result = await self.backend.predict(input)
                    self.metrics.mark_checkpoint("model_complete")
                    self.determine_token_usage(result, input)

                    if MODEL_CALLS_TOTAL:
                        MODEL_CALLS_TOTAL.add(
                            1, {"model_id": self.model_id, "status": "success"}
                        )

                    span.set_status(trace.StatusCode.OK)
                    # Add result metadata to the parent span for context
                    if self.metrics._current_span:
                        self.metrics._current_span.set_attribute(
                            "llm.response.output", str(getattr(result, "output", ""))
                        )

                    return result
                except Exception as e:
                    if MODEL_CALLS_TOTAL:
                        MODEL_CALLS_TOTAL.add(
                            1, {"model_id": self.model_id, "status": "failure"}
                        )
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
        else:
            # Fallback for when OTel is disabled, but preserve original metrics logic
            self.metrics.mark_checkpoint("model_start")
            result = await self.backend.predict(input)
            self.metrics.mark_checkpoint("model_complete")
            self.determine_token_usage(result, input)
            return result

    def get_lm_history(self) -> list[Any]:
        """Pass through to the wrapped backend's get_lm_history method."""
        if hasattr(self.backend, "get_lm_history"):
            return self.backend.get_lm_history()
        return []

    # ALL ORIGINAL, ROBUST TOKEN USAGE LOGIC IS PRESERVED
    def _get_tokens_from_dspy_history(self) -> tuple[int, int] | None:
        if not hasattr(self.backend, "get_lm_history"):
            return None
        try:
            history = self.backend.get_lm_history()
            if not history:
                return None
            last_call = history[-1]
            usage = last_call.get("usage", {})
            prompt_tokens = last_call.get("prompt_tokens") or usage.get("prompt_tokens")
            completion_tokens = last_call.get("completion_tokens") or usage.get(
                "completion_tokens"
            )
            if prompt_tokens is not None and completion_tokens is not None:
                return prompt_tokens, completion_tokens
        except Exception as e:
            logging.warning(f"Could not get tokens from DSPy history: {e}")
        return None

    def _get_tokens_from_result_metadata(self, result: Any) -> tuple[int, int] | None:
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            usage = result.metadata.get("usage", {})
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                return usage["prompt_tokens"], usage["completion_tokens"]
        return None

    def _get_tokens_from_estimation(
        self, input_data: Any, result: Any
    ) -> tuple[int, int]:
        input_str = str(input_data)
        output_str = getattr(result, "output", "")
        estimated_input = len(input_str) // 3
        estimated_output = len(str(output_str)) // 3
        logging.info("Token usage estimated from character count.")
        return estimated_input, estimated_output

    def determine_token_usage(self, result: Any, input_data: Any):
        strategies = [
            (self._get_tokens_from_dspy_history, "dspy_history_exact"),
            (
                lambda: self._get_tokens_from_result_metadata(result),
                "response_metadata_exact",
            ),
        ]
        for strategy_func, method_name in strategies:
            tokens = strategy_func()
            if tokens is not None:
                self.metrics.record_token_usage(
                    input_tokens=tokens[0], output_tokens=tokens[1]
                )
                self.metrics.add_metadata("token_count_method", method_name)
                return
        est_input, est_output = self._get_tokens_from_estimation(input_data, result)
        self.metrics.record_token_usage(
            input_tokens=est_input, output_tokens=est_output
        )
        self.metrics.add_metadata("token_count_method", "character_based_estimate")


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