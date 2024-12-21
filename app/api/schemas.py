from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class ModelConfig(BaseModel):
    model_name: str
    max_tokens: Optional[int] = 1000
    additional_params: Dict[str, Any] = {}
    
    # model_config allows configuring Pydantic model behavior
    model_config = {
        # disable protected namespace checks to allow the model_name field
        'protected_namespaces': ()
    }

class QueryRequest(BaseModel):
    prompt: str
    model_id: str
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, gt=0)
    
    # model_config allows configuring Pydantic model behavior
    model_config = {
        # disable protected namespace checks to allow the model_id field
        'protected_namespaces': ()
    }

class QueryResponse(BaseModel):
    response: str
    model_used: str
    metadata: Dict[str, Any]
    
    # model_config allows configuring Pydantic model behavior
    model_config = {
        # disable protected namespace checks to allow the model_used field
        'protected_namespaces': ()
    }