import pydantic
from typing import Dict, Any, Optional

# Base Models
class ModelConfig(pydantic.BaseModel):
    model_name: str
    max_tokens: Optional[int] = 1000
    additional_params: Dict[str, Any] = {}
    
    model_config = pydantic.ConfigDict(protected_namespaces=())
