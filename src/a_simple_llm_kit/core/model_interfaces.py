from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelOutput(Protocol):
    """Protocol for standardizing model outputs"""

    def to_response(self) -> Any:
        """Convert model output to API response format"""
        ...


class SimpleOutput(ModelOutput):
    """A simple wrapper for string outputs to conform to ModelOutput."""

    def __init__(self, text: str):
        self.output = text

    def to_response(self) -> Any:
        return self.output
