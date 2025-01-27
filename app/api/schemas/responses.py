from datetime import datetime, timezone
from typing import Dict, Any, Generic, Optional, TypeVar
import pydantic
from app.core.types import MediaType
from app.core.modules import ExtractContact

T = TypeVar('T')

class HealthResponse(pydantic.BaseModel):
    status: str

class StandardResponse(pydantic.BaseModel, Generic[T]):
    """Standard envelope for all API responses"""
    success: bool = pydantic.Field(description="Indicates if the request was successful")
    data: Optional[T] = pydantic.Field(default=None, description="The response payload")
    error: Optional[str] = pydantic.Field(default=None, description="Error message if success is false")
    timestamp: datetime = pydantic.Field(default_factory=lambda: datetime.now(timezone.utc))

class QueryResponseData(pydantic.BaseModel):
    """Data structure for query responses"""
    response: str
    model_used: str
    metadata: Dict[str, Any]

class QueryResponse(StandardResponse[QueryResponseData]):
    """Standardized response for query endpoints"""
    pass

class PipelineResponseData(pydantic.BaseModel):
    """Data structure for pipeline responses"""
    content: Any
    media_type: MediaType
    metadata: Dict[str, Any]

class PipelineResponse(StandardResponse[PipelineResponseData]):
    """Standardized response for pipeline endpoints"""
    pass

class ExtractContactResponse(StandardResponse[ExtractContact]):
    """Standardized response for business card data"""
    pass
