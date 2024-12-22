from typing import List
from abc import ABC, abstractmethod

from app.core.types import MediaType, PipelineData

class PipelineStep(ABC):
    """Base class for all pipeline steps."""
    @abstractmethod
    async def process(self, data: PipelineData) -> PipelineData:
        """Process the input data and return transformed output."""
        pass

    @property
    @abstractmethod
    def accepted_media_types(self) -> List[MediaType]:
        """List of media types this step can process."""
        pass

class Pipeline:
    """Manages the execution of multiple pipeline steps."""
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps
        self._validate_pipeline()
    
    def _validate_pipeline(self):
        """Validate that pipeline steps can be connected."""
        for i in range(len(self.steps) - 1):
            current_step = self.steps[i]
            next_step = self.steps[i + 1]
            if not any(media_type in next_step.accepted_media_types 
                      for media_type in current_step.accepted_media_types):
                raise ValueError(
                    f"Incompatible steps: {current_step.__class__.__name__} "
                    f"-> {next_step.__class__.__name__}"
                )
    
    async def execute(self, initial_data: PipelineData) -> PipelineData:
        current_data = initial_data
        
        for step in self.steps:
            if current_data.media_type not in step.accepted_media_types:
                raise ValueError(
                    f"Step {step.__class__.__name__} cannot process "
                    f"media type {current_data.media_type}"
                )
            current_data = await step.process(current_data)
        
        return current_data