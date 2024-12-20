# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import yaml
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import dspy
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# Configuration Models
class ModelConfig(BaseModel):
    model_name: str
    max_tokens: Optional[int] = 1000
    additional_params: Dict[str, Any] = {}

class ServerConfig(BaseModel):
    models: Dict[str, ModelConfig]

# Request/Response Models
class QueryRequest(BaseModel):
    prompt: str
    model_id: str  # Now required
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
        api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_API_KEY')
        }
        
        for model_id, model_config in self.config.models.items():
            try:
                provider = model_config.model_name.split('/')[0]
                api_key = api_keys.get(provider)
                
                if not api_key:
                    raise ValueError(f"API key not found for provider: {provider}")
                
                lm = dspy.LM(
                    model_config.model_name,
                    api_key=api_key,
                    max_tokens=model_config.max_tokens,
                    **model_config.additional_params
                )
                self.models[model_id] = lm
                logging.info(f"Successfully initialized model: {model_id}")
            except Exception as e:
                logging.error(f"Failed to initialize model {model_id}: {str(e)}")
                raise

    @contextmanager
    def get_model(self, model_id: str):
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        try:
            yield self.models[model_id]
        except Exception as e:
            logging.error(f"Error with model {model_id}: {str(e)}")
            raise

class Predictor(dspy.Signature):
    """Simple prediction signature for basic completion."""
    input = dspy.InputField()
    output = dspy.OutputField()

# FastAPI Application
app = FastAPI()
model_manager = None

# Create lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    model_manager = ModelManager("config.yaml")
    
    yield
    # Shutdown: Clean up resources if needed
    model_manager = None

# FastAPI Application with lifespan
app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=QueryResponse)
async def predict(request: QueryRequest):
    with model_manager.get_model(request.model_id) as lm:
        dspy.configure(lm=lm)
        try:
            predictor = dspy.Predict(Predictor, lm)
            result = predictor(input=request.prompt)
            
            return QueryResponse(
                response=result.output,
                model_used=request.model_id,
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