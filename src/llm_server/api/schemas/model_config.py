from typing import Any, Optional

import pydantic


# Base Models
class ModelConfig(pydantic.BaseModel):
    model_name: str
    max_tokens: Optional[int] = 1000
    additional_params: dict[str, Any] = {}

    model_config = pydantic.ConfigDict(protected_namespaces=())
