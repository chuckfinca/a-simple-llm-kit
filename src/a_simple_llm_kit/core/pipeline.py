from collections.abc import Sequence

from a_simple_llm_kit.core.opentelemetry_integration import (
    _OTEL_ENABLED,
    _tracer,
    trace,
)
from a_simple_llm_kit.core.protocols import PipelineStep
from a_simple_llm_kit.core.types import PipelineData


class PipelineValidator:
    """Dedicated validator for pipeline configurations"""

    @staticmethod
    def validate_steps(steps: list[PipelineStep]) -> None:
        """Validate that steps can be connected in sequence"""
        if not steps:
            raise ValueError("Pipeline must contain at least one step")

        for i in range(len(steps) - 1):
            current = steps[i]
            next_step = steps[i + 1]
            if not any(
                media_type in next_step.accepted_media_types
                for media_type in current.accepted_media_types
            ):
                raise ValueError(
                    f"Incompatible steps: {current.__class__.__name__} "
                    f"-> {next_step.__class__.__name__}"
                )

    @staticmethod
    def validate_initial_data(data: PipelineData, first_step: PipelineStep) -> None:
        """Validate initial data compatibility with first step"""
        if data.media_type not in first_step.accepted_media_types:
            raise ValueError(
                f"Initial data type {data.media_type} not compatible with "
                f"first step {first_step.__class__.__name__}"
            )


class Pipeline:
    """Manages execution of multiple pipeline steps in sequence"""

    def __init__(self, steps: Sequence[PipelineStep]):
        self.steps = steps
        self.validator = PipelineValidator()
        self.validator.validate_steps(list(steps))

    async def execute(self, initial_data: PipelineData) -> PipelineData:
        """Execute steps in sequence"""
        self.validator.validate_initial_data(initial_data, self.steps[0])

        if not (_OTEL_ENABLED and _tracer):
            # Fallback to non-traced execution if OTel is disabled
            current_data = initial_data
            for step in self.steps:
                current_data = await step.process(current_data)
            return current_data

        # Execute with tracing using the idiomatic nested `with` pattern
        with _tracer.start_as_current_span(
            "pipeline.execute", attributes={"pipeline.step_count": len(self.steps)}
        ) as parent_span:
            current_data = initial_data
            for i, step in enumerate(self.steps):
                step_name = step.__class__.__name__
                with _tracer.start_as_current_span(
                    f"pipeline.step.{step_name}", attributes={"pipeline.step.index": i}
                ) as step_span:
                    try:
                        # Add rich attributes before processing
                        step_span.set_attribute(
                            "pipeline.step.input_type", current_data.media_type.value
                        )

                        result = await step.process(current_data)

                        # Add attributes after processing
                        step_span.set_attribute(
                            "pipeline.step.output_type", result.media_type.value
                        )
                        current_data = result

                    except Exception as e:
                        step_span.record_exception(e)
                        step_span.set_status(trace.StatusCode.ERROR, str(e))
                        parent_span.set_status(
                            trace.StatusCode.ERROR, f"Step {i} ({step_name}) failed"
                        )
                        raise

            return current_data
