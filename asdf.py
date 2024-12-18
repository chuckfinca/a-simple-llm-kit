# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import dspy
from contextlib import contextmanager
import yaml
import logging
from datetime import datetime

# Configuration Models
class ModelConfig(BaseModel):
    provider: str
    model_name: str
    api_key: Optional[str] = None
    additional_params: Dict[str, Any] = {}

class ServerConfig(BaseModel):
    default_model: str
    models: Dict[str, ModelConfig]

# Request/Response Models
class QueryRequest(BaseModel):
    prompt: str
    model_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class QueryResponse(BaseModel):
    response: str
    model_used: str
    metadata: Dict[str, Any]

class ModelManager:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = ServerConfig(**config)
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        for model_id, model_config in self.config.models.items():
            try:
                if model_config.provider == "modal":
                    self._setup_modal_model(model_id, model_config)
                elif model_config.provider == "huggingface":
                    self._setup_hf_model(model_id, model_config)
                # Easy to add more providers here
            except Exception as e:
                logging.error(f"Failed to initialize model {model_id}: {str(e)}")
                raise

    def _setup_modal_model(self, model_id: str, config: ModelConfig):
        # Modal-specific setup
        try:
            import modal
            stub = modal.Stub(f"llm-server-{model_id}")
            
            @stub.function()
            def predict(prompt: str, **kwargs):
                # Modal-specific prediction implementation
                pass
            
            self.models[model_id] = predict
        except ImportError:
            logging.error("Modal package not found. Please install modal-client")
            raise

    def _setup_hf_model(self, model_id: str, config: ModelConfig):
        # HuggingFace-specific setup
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # HF-specific implementation
            pass
        except ImportError:
            logging.error("Transformers package not found. Please install transformers")
            raise

    @contextmanager
    def get_model(self, model_id: Optional[str] = None):
        model_id = model_id or self.config.default_model
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        try:
            yield self.models[model_id]
        except Exception as e:
            logging.error(f"Error with model {model_id}: {str(e)}")
            raise

# FastAPI Application
app = FastAPI()
model_manager = None

@app.on_event("startup")
async def startup_event():
    global model_manager
    model_manager = ModelManager("config.yaml")

@app.post("/predict", response_model=QueryResponse)
async def predict(request: QueryRequest):
    with model_manager.get_model(request.model_id) as model:
        try:
            response = model(
                request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return QueryResponse(
                response=response,
                model_used=request.model_id or model_manager.config.default_model,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)