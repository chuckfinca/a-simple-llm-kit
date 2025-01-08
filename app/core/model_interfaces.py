from typing import Any, Optional, Protocol, runtime_checkable
import pydantic

@runtime_checkable
class ModelOutput(Protocol):
    """Protocol for standardizing model outputs"""
    def to_response(self) -> Any:
        """Convert model output to API response format"""
        ...

@runtime_checkable
class Signature(Protocol):
    """Protocol for model signatures"""
    @classmethod
    def process_output(cls, result: Any) -> ModelOutput:
        """Process raw model output into standardized format"""
        ...
