from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel

class MediaType(Enum):
    TEXT = "text"
    IMAGE = "image"

class PipelineData(BaseModel):
    """Container for data passing through pipeline steps"""
    media_type: MediaType
    content: Any
    metadata: Dict[str, Any] = {}

