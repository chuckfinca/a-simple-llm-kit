from enum import Enum
from typing import Dict, Any
import pydantic

class MediaType(Enum):
    TEXT = "text"
    IMAGE = "image"

class PipelineData(pydantic.BaseModel):
    """Container for data passing through pipeline steps"""
    media_type: MediaType
    content: Any
    metadata: Dict[str, Any] = {}

