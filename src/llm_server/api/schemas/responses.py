from datetime import datetime
from typing import Dict, Any, Generic, Optional, TypeVar, Union
import pydantic
from llm_server.core.types import MediaType
from llm_server.core.modules import ExtractContact
from llm_server.core.utils import get_utc_now

T = TypeVar('T')

class ModelInfo(pydantic.BaseModel):
    """Structured model information"""
    id: str
    provider: Optional[str] = None
    base_name: Optional[str] = None
    full_name: Optional[str] = None

class ProgramInfo(pydantic.BaseModel):
    """Structured program information"""
    id: str
    version: str
    name: Optional[str] = None

class PerformanceInfo(pydantic.BaseModel):
    """Structured performance metrics"""
    total_ms: float
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    trace_id: Optional[str] = None
    cost_usd: Optional[float] = None

class HealthResponse(pydantic.BaseModel):
    status: str

class QueryResponseData(pydantic.BaseModel):
    """Data structure for query responses - streamlined to just contain the response"""
    response: str

class StandardResponse(pydantic.BaseModel, Generic[T]):
    """Standard envelope for all API responses"""
    success: bool = pydantic.Field(description="Indicates if the request was successful")
    data: Optional[T] = pydantic.Field(default=None, description="The response payload")
    error: Optional[Union[str, Dict[str, Any]]] = pydantic.Field(default=None, description="Error message or object if success is false")
    timestamp: datetime = pydantic.Field(default_factory=get_utc_now)
    metadata: Dict[str, Any] = pydantic.Field(default_factory=dict, description="Response metadata including program/model info")

class QueryResponse(StandardResponse[QueryResponseData]):
    """Standardized response for query endpoints"""
    pass

class PipelineResponseData(pydantic.BaseModel):
    """Data structure for pipeline responses"""
    content: Any
    media_type: MediaType
    metadata: Dict[str, Any] = pydantic.Field(default_factory=dict)  # Empty dict by default

class PipelineResponse(StandardResponse[PipelineResponseData]):
    """Standardized response for pipeline endpoints"""
    pass

class ExtractContactResponse(StandardResponse[ExtractContact]):
    """Standardized response for contact data"""
    pass