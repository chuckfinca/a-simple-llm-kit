from typing import List

from app.core.protocols import PipelineStep
from app.core.types import PipelineData

class PipelineValidator:
    """Dedicated validator for pipeline configurations"""
    @staticmethod
    def validate_steps(steps: List[PipelineStep]) -> None:
        """Validate that steps can be connected in sequence"""
        if not steps:
            raise ValueError("Pipeline must contain at least one step")
            
        for i in range(len(steps) - 1):
            current = steps[i]
            next_step = steps[i + 1]
            if not any(media_type in next_step.accepted_media_types 
                      for media_type in current.accepted_media_types):
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
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps
        self.validator = PipelineValidator()
        self.validator.validate_steps(steps)
    
    async def execute(self, initial_data: PipelineData) -> PipelineData:
        """Execute steps in sequence"""
        self.validator.validate_initial_data(initial_data, self.steps[0])
        
        current_data = initial_data
        for step in self.steps:
            current_data = await step.process(current_data)
        return current_data
