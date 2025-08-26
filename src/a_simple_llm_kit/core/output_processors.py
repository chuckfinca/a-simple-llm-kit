import json
from typing import Any

from a_simple_llm_kit.core import logging
from a_simple_llm_kit.core.protocols import OutputProcessor
from a_simple_llm_kit.core.types import PipelineData


class DefaultOutputProcessor(OutputProcessor):
    def process(self, result: Any, pipeline_data: PipelineData) -> Any:
        if hasattr(result, "output"):
            return result.output
        # Also handle cases where the result might be from a TypedPredictor
        if hasattr(result, "model_dump_json"):
            try:
                # It's a Pydantic model
                return json.loads(result.model_dump_json())
            except Exception:
                pass  # Fallback to returning the object
        logging.warning(
            "DefaultOutputProcessor: Result has no 'output' attribute and is not a standard Pydantic model."
        )
        return result
