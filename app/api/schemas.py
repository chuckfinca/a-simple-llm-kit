from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, Union
from app.core.types import MediaType

class ModelConfig(BaseModel):
    model_name: str
    max_tokens: Optional[int] = 1000
    additional_params: Dict[str, Any] = {}
    
    model_config = ConfigDict(protected_namespaces=())

class QueryRequest(BaseModel):
    prompt: str
    model_id: str
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, gt=0)
    
    model_config = ConfigDict(protected_namespaces=())

class QueryResponse(BaseModel):
    response: str
    model_used: str
    metadata: Dict[str, Any]
    
    model_config = ConfigDict(protected_namespaces=())
    
class PipelineRequest(BaseModel):
    """Generic request schema for pipeline processing."""
    pipeline_id: str
    content: Union[str, bytes]  # Text content or base64 encoded image
    media_type: MediaType
    params: Dict[str, Any] = {}

class PipelineResponse(BaseModel):
    """Generic response schema for pipeline results."""
    content: Any
    media_type: MediaType
    metadata: Dict[str, Any]