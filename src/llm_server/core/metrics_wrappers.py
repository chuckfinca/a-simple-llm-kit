import time
import uuid
from typing import Any, Optional

from llm_server.src.llm_server.core import logging
from contactcapture_backend.core.protocols import ModelBackend, PipelineStep
from contactcapture_backend.core.types import MediaType, PipelineData


class PerformanceMetrics:
    """Tracks performance metrics throughout request processing"""

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

    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage information"""
        self.token_usage["input"] = input_tokens
        self.token_usage["output"] = output_tokens
        self.token_usage["total"] = input_tokens + output_tokens

        # Store in metadata too
        self.metadata["input_tokens"] = input_tokens
        self.metadata["output_tokens"] = output_tokens
        self.metadata["total_tokens"] = input_tokens + output_tokens

        # Calculate cost if we have the model info
        if self.model_id:
            self._calculate_cost()

    def set_model_info(
        self, model_id: str, model_info: Optional[dict[str, Any]] = None
    ) -> None:
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
        # Cost rates per 1M tokens (input, output) in USD as of March 24, 2025
        pricing = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.5, 10),
            "openai-o3-mini": (1.1, 4.4),
            "claude-3.7-sonnet": (3.0, 15.0),
            "claude-3.5-haiku": (0.8, 4.0),
            "gemini-2.0-flash": (0.1, 0.4),
            "gemini-2.0-flash-lite": (0.075, 0.3),
            # Fallback rates for unknown models
            "default": (0.0015, 0.002),
        }

        # Get rate for the model, default to fallback rate if not found
        key = self.model_id if self.model_id else "default"
        input_rate, output_rate = pricing.get(key, pricing["default"])

        # Calculate cost
        input_cost = (self.token_usage["input"] / 1000) * input_rate
        output_cost = (self.token_usage["output"] / 1000) * output_rate
        total_cost = input_cost + output_cost

        # Store in metadata
        self.metadata["estimated_cost_usd"] = round(total_cost, 6)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the metrics for API responses"""
        # Calculate total time
        total_time = time.time() - self.start_time

        # Create tokens structure with method information and cost
        tokens_data = self.token_usage.copy()

        # Add the token calculation method inside the tokens object
        if "token_count_method" in self.metadata:
            tokens_data["method"] = self.metadata["token_count_method"]

        # Add cost inside the tokens object
        if "estimated_cost_usd" in self.metadata:
            tokens_data["cost_usd"] = self.metadata["estimated_cost_usd"]

        # Create summary with basic timing information
        summary = {
            "timing": {
                "total_ms": round(total_time * 1000, 2),
            },
            "tokens": tokens_data,
            "trace_id": self.trace_id,
        }

        # Add detailed timing for each checkpoint
        for name, timestamp in self.checkpoints.items():
            if name != "request_start":
                summary["timing"][f"{name}_ms"] = round(
                    (timestamp - self.start_time) * 1000, 2
                )

        # Add pipeline step timing if available
        if "step_timing" in self.metadata:
            summary["step_timing"] = self.metadata["step_timing"]

            # Calculate step percentages of total time
            if total_time > 0:
                step_percentages = {}
                for step_name, timing in self.metadata["step_timing"].items():
                    step_duration = (
                        timing.get("duration_ms", 0) / 1000
                    )  # Convert back to seconds
                    step_percentages[step_name] = round(
                        (step_duration / total_time) * 100, 1
                    )
                summary["step_percentages"] = step_percentages

        # Add steps in sequence if available
        if "pipeline_steps" in self.metadata:
            summary["pipeline_steps"] = self.metadata["pipeline_steps"]

        return summary

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics for logging"""
        all_metrics = {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "total_time": time.time() - self.start_time,
            "checkpoints": self.checkpoints.copy(),
            "token_usage": self.token_usage.copy(),
            "model_id": self.model_id,
        }

        # Add all metadata
        all_metrics.update(self.metadata)

        return all_metrics


class ModelBackendTracker:
    """
    Wrapper that adds performance tracking to any ModelBackend implementation
    using composition rather than inheritance
    """

    def __init__(
        self, backend: ModelBackend, metrics: Optional[PerformanceMetrics] = None
    ):
        """
        Initialize tracker with a backend and optional metrics instance

        Args:
            backend: The ModelBackend to wrap
            metrics: Optional PerformanceMetrics instance (creates new one if None)
        """
        self.backend = backend
        self.metrics = metrics or PerformanceMetrics()

        # Initialize token tracking variables
        self.pre_prompt_tokens = 0
        self.pre_completion_tokens = 0

        self.last_prompt_tokens: Optional[int] = None
        self.last_completion_tokens: Optional[int] = None

        # Pass through model ID if available
        if hasattr(backend, "model_id"):
            self.model_id = backend.model_id
            self.metrics.set_model_info(backend.model_id)

        # Directly expose program_metadata from backend for versioning middleware
        if hasattr(backend, "program_metadata"):
            self.program_metadata = backend.program_metadata

    async def predict(self, input: Any) -> Any:
        """Execute prediction with metrics tracking"""
        # Mark preparation complete
        self.metrics.mark_checkpoint("preparation_complete")

        # Store initial token counts for diff calculation if available
        self.pre_prompt_tokens = getattr(self.backend, "total_prompt_tokens", 0)
        self.pre_completion_tokens = getattr(self.backend, "total_completion_tokens", 0)

        # Mark model start
        self.metrics.mark_checkpoint("model_start")

        try:
            # Execute prediction on wrapped backend
            result = await self.backend.predict(input)

            # Mark model complete
            self.metrics.mark_checkpoint("model_complete")

            # Record token usage using the new encapsulated method
            self.determine_token_usage(result, input)

            # Add metrics to result if possible
            if hasattr(result, "metadata"):
                # Create a new dict if needed
                if not isinstance(result.metadata, dict):
                    result.metadata = {}

                # Add metrics summary to metadata
                result.metadata["performance_metrics"] = self.metrics.get_summary()

                # Also add program_metadata if available
                if hasattr(self.backend, "program_metadata"):
                    result.metadata["program_metadata"] = self.backend.program_metadata

            return result

        except Exception as e:
            # Mark error in metrics
            self.metrics.add_metadata("error", str(e))
            self.metrics.add_metadata("error_type", type(e).__name__)
            self.metrics.mark_checkpoint("error")

            # Log the error with metrics context
            logging.error(
                f"Error in model execution: {str(e)}",
                extra={"performance_metrics": self.metrics.get_all_metrics()},
            )

            # Re-raise the exception
            raise

    def get_lm_history(self) -> list[Any]:
        """Pass through to the wrapped backend's get_lm_history method."""
        if hasattr(self.backend, "get_lm_history"):
            return self.backend.get_lm_history()
        return []

        # --- Token Usage Strategy Methods ---

    def _get_tokens_from_dspy_history(self) -> Optional[tuple[int, int]]:
        """Strategy 1: Get token usage from DSPy LM history."""
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
                logging.info(
                    f"Token usage from DSPy history: {prompt_tokens} input, {completion_tokens} output"
                )
                return prompt_tokens, completion_tokens
        except Exception as e:
            logging.warning(f"Could not get tokens from DSPy history: {e}")
        return None

    def _get_tokens_from_result_metadata(
        self, result: Any
    ) -> Optional[tuple[int, int]]:
        """Strategy 2: Get token usage from the result object's metadata."""
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            usage = result.metadata.get("usage", {})
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                return usage["prompt_tokens"], usage["completion_tokens"]
        return None

    def _get_tokens_from_backend_attributes(self) -> Optional[tuple[int, int]]:
        """Strategy 3: Get token usage from direct attributes on the backend."""
        prompt_tokens = getattr(self.backend, "last_prompt_tokens", None)
        completion_tokens = getattr(self.backend, "last_completion_tokens", None)

        # Only return a tuple if BOTH values are valid integers
        if prompt_tokens is not None and completion_tokens is not None:
            return prompt_tokens, completion_tokens

        # Otherwise, return None
        return None

    def _get_tokens_from_estimation(
        self, input_data: Any, result: Any
    ) -> tuple[int, int]:
        """Strategy 4 (Fallback): Estimate tokens based on character count."""
        input_str = str(input_data)
        output_str = getattr(result, "output", "")
        estimated_input = len(input_str) // 3
        estimated_output = len(str(output_str)) // 3
        logging.info("Token usage estimated from character count.")
        return estimated_input, estimated_output

    # --- Main Method ---

    def determine_token_usage(self, result: Any, input_data: Any):
        """
        Records token usage by trying a sequence of strategies.
        """
        strategies = [
            (self._get_tokens_from_dspy_history, "dspy_history_exact"),
            (
                lambda: self._get_tokens_from_result_metadata(result),
                "response_metadata_exact",
            ),
            (self._get_tokens_from_backend_attributes, "backend_attributes_exact"),
        ]

        # Try exact strategies first
        for strategy_func, method_name in strategies:
            tokens = strategy_func()
            if tokens is not None:
                self.metrics.record_token_usage(
                    input_tokens=tokens[0], output_tokens=tokens[1]
                )
                self.metrics.add_metadata("token_count_method", method_name)
                return

        # Fallback to estimation if no exact method works
        est_input, est_output = self._get_tokens_from_estimation(input_data, result)
        self.metrics.record_token_usage(
            input_tokens=est_input, output_tokens=est_output
        )
        self.metrics.add_metadata("token_count_method", "character_based_estimate")


class PipelineStepTracker:
    """
    Wrapper that adds performance tracking to any PipelineStep implementation
    using composition rather than inheritance
    """

    def __init__(
        self,
        step: PipelineStep,
        metrics: Optional[PerformanceMetrics] = None,
        step_name: Optional[str] = None,
    ):
        """
        Initialize tracker with a pipeline step and optional metrics instance

        Args:
            step: The PipelineStep to wrap
            metrics: Optional PerformanceMetrics instance (creates new one if None)
            step_name: Optional name for this step (defaults to class name)
        """
        self.step = step
        self.metrics = metrics or PerformanceMetrics()
        self.step_name = step_name or step.__class__.__name__

    @property
    def accepted_media_types(self) -> list[MediaType]:
        """Pass through to wrapped step"""
        return self.step.accepted_media_types

    async def process(self, data: PipelineData) -> PipelineData:
        """
        Process data with metrics tracking

        Args:
            data: The PipelineData to process

        Returns:
            Processed PipelineData from the wrapped step
        """
        # Mark step start with more detailed naming
        step_start_key = f"{self.step_name}_start"
        self.metrics.mark_checkpoint(step_start_key)

        try:
            # Execute the wrapped step
            result = await self.step.process(data)

            # Mark step complete
            step_complete_key = f"{self.step_name}_complete"
            self.metrics.mark_checkpoint(step_complete_key)

            # Calculate and record step duration explicitly
            start_time = self.metrics.checkpoints.get(step_start_key)
            complete_time = self.metrics.checkpoints.get(step_complete_key)
            if start_time and complete_time:
                duration = complete_time - start_time
                self.metrics.add_metadata(
                    f"{self.step_name}_duration_ms", round(duration * 1000, 2)
                )

                # Create new metadata dictionary that includes our metrics
                # Include per-step timing in the metrics
                step_metrics = {
                    "step_timing": self.metrics.metadata.get("step_timing", {})
                }

                # Add this step's timing to the step_timing dictionary
                step_metrics["step_timing"][self.step_name] = {
                    "start_ms": round((start_time - self.metrics.start_time) * 1000, 2),
                    "end_ms": round(
                        (complete_time - self.metrics.start_time) * 1000, 2
                    ),
                    "duration_ms": round(duration * 1000, 2),
                }

                # Update metrics' metadata with the step timing
                self.metrics.metadata["step_timing"] = step_metrics["step_timing"]

            # Add step-specific metadata to metrics
            self.metrics.add_metadata(
                "pipeline_steps",
                self.metrics.metadata.get("pipeline_steps", []) + [self.step_name],
            )

            # Add step-specific metrics if available
            if (
                self.step_name == "ImageProcessor"
                and "original_size" in result.metadata
            ):
                self.metrics.add_metadata(
                    "image_original_size", result.metadata["original_size"]
                )
                self.metrics.add_metadata(
                    "image_processed_size", result.metadata.get("processed_size")
                )

            # Combine with result metadata
            combined_metadata = {**result.metadata}

            # Add performance metrics with a consistent key
            # Use "performance_metrics" key to match middleware expectations
            combined_metadata["performance_metrics"] = self.metrics.get_summary()

            # Return new PipelineData with updated metadata
            return PipelineData(
                media_type=result.media_type,
                content=result.content,
                metadata=combined_metadata,
            )

        except Exception as e:
            # Mark error in metrics
            step_error_key = f"{self.step_name}_error"
            self.metrics.add_metadata("error", str(e))
            self.metrics.add_metadata("error_type", type(e).__name__)
            self.metrics.add_metadata("error_step", self.step_name)
            self.metrics.mark_checkpoint(step_error_key)

            # Log the error with metrics context
            logging.error(
                f"Error in {self.step_name}: {str(e)}",
                extra={"performance_metrics": self.metrics.get_all_metrics()},
            )

            # Re-raise the exception
            raise


class TrackedPipeline:
    """
    A pipeline that tracks performance metrics for all steps
    """

    def __init__(self, pipeline, metrics: Optional[PerformanceMetrics] = None):
        """
        Initialize with a pipeline and optional metrics

        Args:
            pipeline: The Pipeline to wrap
            metrics: Optional PerformanceMetrics instance (creates new one if None)
        """
        self.pipeline = pipeline
        self.metrics = metrics or PerformanceMetrics()
        # Track the step count for diagnostics
        self.step_count = len(getattr(pipeline, "steps", []))
        self.metrics.add_metadata("total_pipeline_steps", self.step_count)

    async def execute(self, data: PipelineData) -> PipelineData:
        """
        Execute the pipeline with performance tracking

        Args:
            data: Initial pipeline data

        Returns:
            Result from pipeline execution
        """
        # Add metrics to data metadata so steps can access it
        initial_data = PipelineData(
            media_type=data.media_type,
            content=data.content,
            metadata={**data.metadata, "metrics_collector": self.metrics},
        )

        # Mark pipeline start
        self.metrics.mark_checkpoint("pipeline_start")

        try:
            # Execute the pipeline
            result = await self.pipeline.execute(initial_data)

            # Mark pipeline complete
            self.metrics.mark_checkpoint("pipeline_complete")

            # Calculate pipeline execution time
            pipeline_start = self.metrics.checkpoints.get("pipeline_start")
            pipeline_complete = self.metrics.checkpoints.get("pipeline_complete")

            if pipeline_start and pipeline_complete:
                pipeline_duration = pipeline_complete - pipeline_start
                self.metrics.add_metadata(
                    "pipeline_duration_ms", round(pipeline_duration * 1000, 2)
                )

            # Aggregate step timing statistics if available
            if "step_timing" in self.metrics.metadata:
                step_stats = {
                    "min_duration_ms": float("inf"),
                    "max_duration_ms": 0,
                    "avg_duration_ms": 0,
                    "slowest_step": None,
                    "fastest_step": None,
                }

                total_step_time = 0
                step_count = 0

                for step_name, timing in self.metrics.metadata["step_timing"].items():
                    duration_ms = timing.get("duration_ms", 0)
                    total_step_time += duration_ms
                    step_count += 1

                    if duration_ms > step_stats["max_duration_ms"]:
                        step_stats["max_duration_ms"] = duration_ms
                        step_stats["slowest_step"] = step_name

                    if duration_ms < step_stats["min_duration_ms"]:
                        step_stats["min_duration_ms"] = duration_ms
                        step_stats["fastest_step"] = step_name

                if step_count > 0:
                    step_stats["avg_duration_ms"] = round(
                        total_step_time / step_count, 2
                    )
                    self.metrics.add_metadata("step_statistics", step_stats)

            # Add metrics to result metadata
            # Merge with existing metadata without overriding
            combined_metadata = result.metadata.copy()

            # Only add performance_metrics if not already present
            if "performance_metrics" not in combined_metadata:
                combined_metadata["performance_metrics"] = self.metrics.get_summary()

            # Create new PipelineData with updated metadata
            return PipelineData(
                media_type=result.media_type,
                content=result.content,
                metadata=combined_metadata,
            )

        except Exception as e:
            # Mark error in metrics
            self.metrics.add_metadata("error", str(e))
            self.metrics.add_metadata("error_type", type(e).__name__)
            self.metrics.mark_checkpoint("pipeline_error")

            # Log the error with metrics context
            logging.error(
                f"Error in pipeline execution: {str(e)}",
                extra={"performance_metrics": self.metrics.get_all_metrics()},
            )

            # Re-raise the exception
            raise


class TrackingFactory:
    """
    Factory to create tracking wrappers for various components
    """

    @staticmethod
    def track_backend(
        backend: ModelBackend, metrics: Optional[PerformanceMetrics] = None
    ) -> ModelBackendTracker:
        """Create a tracked backend wrapper"""
        return ModelBackendTracker(backend, metrics)

    @staticmethod
    def track_step(
        step: PipelineStep,
        metrics: Optional[PerformanceMetrics] = None,
        step_name: Optional[str] = None,
    ) -> PipelineStepTracker:
        """Create a tracked pipeline step wrapper"""
        return PipelineStepTracker(step, metrics, step_name)

    @staticmethod
    def track_pipeline(
        pipeline, metrics: Optional[PerformanceMetrics] = None
    ) -> TrackedPipeline:
        """Create a tracked pipeline wrapper"""
        return TrackedPipeline(pipeline, metrics)

    @staticmethod
    def setup_metrics(
        metrics: Optional[PerformanceMetrics], model_id: str, program_manager=None
    ) -> PerformanceMetrics:
        """
        Setup metrics with model information from program_manager if available

        Args:
            metrics: Optional PerformanceMetrics instance (creates a new one if None)
            model_id: The model ID to set
            program_manager: Optional program manager for extracting model info

        Returns:
            Configured PerformanceMetrics instance
        """
        metrics = metrics or PerformanceMetrics()

        # Try to get model info from program_manager
        if (
            program_manager
            and hasattr(program_manager, "model_info")
            and model_id in program_manager.model_info
        ):
            model_info = program_manager.model_info.get(model_id, {})
            metrics.set_model_info(model_id, model_info)
        else:
            metrics.set_model_info(model_id)

        return metrics
