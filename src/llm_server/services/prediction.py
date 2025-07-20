import logging
from typing import Any, Optional, Tuple, Union

import dspy

from llm_server.api.schemas.requests import QueryRequest
from llm_server.models.predictor import Predictor


class PredictionService:
    def __init__(self, model_manager, program_manager=None):
        self.model_manager = model_manager
        self.program_manager = program_manager

    async def predict(
        self, request: Union[QueryRequest, dict[str, Any]]
    ) -> Tuple[str, Optional[Any]]:
        """
        Process a prediction request using the specified model.

        Args:
            request: QueryRequest containing prompt and model configuration,
                    or a dictionary with the same fields

        Returns:
            A tuple of (result, execution_info) where execution_info is None if
            program_manager is not available
        """
        # Extract main parameters, whether from an object or dict
        if isinstance(request, dict):
            prompt = request.get("prompt", "")
            model_id = request.get("model_id", "")
            # Any extra fields will be in the dict already
        else:
            prompt = request.prompt
            model_id = request.model_id
            # Also capture any extra fields that might have been provided
            extra_fields = getattr(request, "model_extra", {})

        # Log any extra parameters for debugging
        if (
            isinstance(request, QueryRequest)
            and hasattr(request, "model_extra")
            and request.model_extra
        ):
            logging.debug(f"Additional parameters received: {request.model_extra}")

        if self.program_manager:
            # Get or register the Predictor program
            # First check if it's already registered
            registered_programs = self.program_manager.registry.list_programs()
            program_id = None

            for prog in registered_programs:
                prog_class = self.program_manager.registry.get_program(prog.id)
                if prog_class and prog_class.__name__ == Predictor.__name__:
                    program_id = prog.id
                    break

            # If not found, register it
            if not program_id:
                metadata = self.program_manager.register_program(
                    program_class=Predictor,
                    name="Text Completion",
                    description="Basic text completion signature",
                )
                program_id = metadata.id

            # Execute with program tracking
            result, execution_info = await self.program_manager.execute_program(
                program_id=program_id, model_id=model_id, input_data={"input": prompt}
            )

            return result.output, execution_info
        else:
            # Fallback to direct execution without tracking
            lm = self.model_manager.get_model(model_id)
            dspy.configure(lm=lm)
            predictor = dspy.Predict(Predictor)
            result = predictor(input=prompt)
            return result.output, None
