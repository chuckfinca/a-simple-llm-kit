from fastapi import APIRouter, HTTPException, Request
from app.api.schemas import PipelineRequest, PipelineResponse, QueryRequest, QueryResponse
from app.core.pipeline import Pipeline
from app.core.types import PipelineData
from app.pipelines.registry import PipelineRegistry
from app.pipelines.steps.text import TextCompletionStep
from app.services.pipelines.predictor import PredictorStep
from app.services.prediction import PredictionService
from datetime import datetime

router = APIRouter()
registry = PipelineRegistry()

@router.post("/predict", response_model=QueryResponse)
async def predict(request: Request, query: QueryRequest):
    try:
        prediction_service = PredictionService(request.app.state.model_manager)
        result = await prediction_service.predict(query)
        
        return QueryResponse(
            response=result,
            model_used=query.model_id,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "temperature": query.temperature,
                "max_tokens": query.max_tokens
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline/predict")
async def predict_pipeline(request: Request, pipeline_req: PipelineRequest):
    # Create predictor pipeline using the app's model manager
    model_manager = request.app.state.model_manager
    pipeline_steps = [TextCompletionStep(model_manager, pipeline_req.params.get("model_id", "gpt-4"))]
    pipeline = Pipeline(pipeline_steps)
    
    # Execute pipeline
    result = await pipeline.execute(PipelineData(
        media_type=pipeline_req.media_type,
        content=pipeline_req.content,
        metadata=pipeline_req.params
    ))
    
    return PipelineResponse(
        content=result.content,
        media_type=result.media_type,
        metadata=result.metadata
    )