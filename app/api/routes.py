import dspy
from fastapi import APIRouter, Depends, HTTPException, Request
from app.api.schemas.requests import PipelineRequest, QueryRequest
from app.api.schemas.responses import BusinessCardResponse, PipelineResponse, QueryResponse
from app.core.pipeline import Pipeline
from app.core.rate_limiting import RateLimit, rate_limit
from app.core.types import PipelineData, MediaType
from app.core.factories import create_business_card_processor, create_text_processor
from app.core.security import get_api_key
from app.services.prediction import PredictionService
from datetime import datetime, timezone

router = APIRouter()

@router.get("/health")
async def health_check(rate_check=None):
    return {"status": "healthy"}

@router.post("/predict", response_model=QueryResponse)
async def predict(
    request: Request, 
    query: QueryRequest, 
    api_key: str = Depends(get_api_key),
    rate_check: None = Depends(rate_limit(RateLimit(
        unauthenticated=5,
        authenticated=100,
        window=30
    )))):
    try:
        prediction_service = PredictionService(request.app.state.model_manager)
        result = await prediction_service.predict(query)
        
        return QueryResponse(
            response=result,
            model_used=query.model_id,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "temperature": query.temperature,
                "max_tokens": query.max_tokens
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline/predict")
async def predict_pipeline(
    request: Request, 
    pipeline_req: PipelineRequest, 
    api_key: str = Depends(get_api_key),
    rate_check: None = Depends(rate_limit(RateLimit(
        unauthenticated=5,
        authenticated=100,
        window=30
    )))):
    model_manager = request.app.state.model_manager
    
    # Use factories to create processors based on media type
    if pipeline_req.media_type == MediaType.TEXT:
        processors = [create_text_processor(model_manager, pipeline_req.params.get("model_id"))]
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
    
@router.post("/pipeline/business-card")
async def process_business_card(
    request: Request, 
    pipeline_req: PipelineRequest, 
    api_key: str = Depends(get_api_key),
    rate_check: None = Depends(rate_limit(RateLimit(
        unauthenticated=5,
        authenticated=100,
        window=30
    )))):
    try:
        model_manager = request.app.state.model_manager
        pipeline = create_business_card_processor(
            model_manager,
            pipeline_req.params.get("model_id"),
        )
        
        result = await pipeline.execute(PipelineData(
            media_type=MediaType.IMAGE,
            content=pipeline_req.content,
            metadata=pipeline_req.params
        ))
        
        dspy.inspect_history(n=1)
        
        return BusinessCardResponse(
            success=True,
            data=result.content,  # This is now a BusinessCard domain model
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))