from typing import Dict, List, Type

from app.core.pipeline import Pipeline, PipelineStep

class PipelineRegistry:
    """Registry for managing different pipeline configurations."""
    def __init__(self):
        self._pipeline_configs: Dict[str, List[Type[PipelineStep]]] = {}
    
    def register(self, pipeline_id: str, steps: List[Type[PipelineStep]]):
        """Register a new pipeline configuration."""
        self._pipeline_configs[pipeline_id] = steps
    
    def create_pipeline(self, pipeline_id: str) -> Pipeline:
        """Create a pipeline instance from registered configuration."""
        if pipeline_id not in self._pipeline_configs:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        steps = [step() for step in self._pipeline_configs[pipeline_id]]
        return Pipeline(steps)