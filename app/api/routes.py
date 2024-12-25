from fastapi import APIRouter, HTTPException, Request
from app.api.schemas import PipelineRequest, PipelineResponse, QueryRequest, QueryResponse
from app.core.pipeline import Pipeline
from app.core.types import PipelineData, MediaType
from app.core.factories import create_text_processor, create_ocr_processor
from app.services.prediction import PredictionService
from datetime import datetime

router = APIRouter()

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
    model_manager = request.app.state.model_manager
    
    # Use factories to create processors based on media type
    if pipeline_req.media_type == MediaType.TEXT:
        processors = [create_text_processor(model_manager, pipeline_req.params.get("model_id"))]
    elif pipeline_req.media_type == MediaType.IMAGE:
        processors = [create_ocr_processor(model_manager, pipeline_req.params.get("model_id"))]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported media type: {pipeline_req.media_type}")
    
    try:
        pipeline = Pipeline(processors)
        
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))