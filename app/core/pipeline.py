from app.core.protocols import Processor
from app.core.types import PipelineData

class Pipeline:
    """Manages execution of multiple processors in sequence"""
    def __init__(self, processors: list[Processor]):
        if not processors:
            raise ValueError("Pipeline must contain at least one processor")
        self.processors = processors
        self._validate_pipeline()
    
    def _validate_pipeline(self):
        """Validate processors can be connected"""
        for i in range(len(self.processors) - 1):
            current = self.processors[i]
            next_proc = self.processors[i + 1]
            if not any(media_type in next_proc.accepted_media_types 
                      for media_type in current.accepted_media_types):
                raise ValueError(
                    f"Incompatible processors: {current.__class__.__name__} "
                    f"-> {next_proc.__class__.__name__}"
                )
    
    async def execute(self, initial_data: PipelineData) -> PipelineData:
        """Execute processors in sequence"""
        current_processor = self.processors[0]
        if initial_data.media_type not in current_processor.accepted_media_types:
            raise ValueError(
                f"Initial data type {initial_data.media_type} not compatible with "
                f"first processor {current_processor.__class__.__name__}"
            )
            
        current_data = initial_data
        for processor in self.processors:
            current_data = await processor.process(current_data)
        return current_data