from typing import Any, Generic, TypeVar, Union

import pydantic

from llm_server.core.types import MediaType

# Generic wrapper for all API requests
T = TypeVar("T")


class RequestWrapper(pydantic.BaseModel, Generic[T]):
    """Generic wrapper for all API requests"""

    request: T


# Request Models
class QueryRequest(pydantic.BaseModel):
    prompt: str
    model_id: str
    temperature: float = pydantic.Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = pydantic.Field(default=1000, gt=0)

    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="allow")


class PipelineRequest(pydantic.BaseModel):
    """Generic request schema for pipeline processing."""

    pipeline_id: str
    content: Union[str, bytes]  # Text content or base64 encoded image
    media_type: MediaType
    params: dict[str, Any] = {}

    model_config = pydantic.ConfigDict(extra="allow")
