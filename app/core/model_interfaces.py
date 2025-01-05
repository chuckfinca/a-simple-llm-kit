from typing import Any, Optional, Protocol, runtime_checkable
from pydantic import BaseModel

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

class TextOutput(BaseModel):
    """Text output from language models"""
    text: str
    
    def to_response(self) -> str:
        return self.text

class BusinessCardOutput(BaseModel):
    """Structured business card data"""
    name: dict
    work: dict
    contact: dict
    notes: Optional[str] = None
    
    def to_response(self) -> dict:
        return self.model_dump()
        
