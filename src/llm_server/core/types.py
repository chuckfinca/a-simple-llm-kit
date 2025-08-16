from enum import Enum
from typing import Any

import pydantic


class MediaType(Enum):
    TEXT = "text"
    IMAGE = "image"


class Usage(pydantic.BaseModel):
    """A generic, provider-agnostic model for token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


class PipelineData(pydantic.BaseModel):
    """Container for data passing through pipeline steps"""

    media_type: MediaType
    content: Any
    metadata: dict[str, Any] = {}


class ProgramMetadata(pydantic.BaseModel):
    """Metadata for a DSPy program signature"""

    id: str
    name: str
    version: str
    code_hash: str
    description: str = ""
    tags: list[str] = []
    parent_id: str | None = None
    parent_version: str | None = None


class ProgramExecutionInfo(pydantic.BaseModel):
    """Information about a specific program execution"""

    program_id: str
    program_version: str
    program_name: str
    model_id: str
    model_info: dict[str, Any] = {}
    execution_id: str
    timestamp: str
    trace_id: str | None = None
