import dspy
from fastapi import APIRouter, Depends, HTTPException, Request
from app.api.schemas.requests import PipelineRequest, QueryRequest
from app.api.schemas.responses import ExtractContactResponse, HealthResponse, PipelineResponse, QueryResponse, QueryResponseData
from app.core import logging
from app.core.pipeline import Pipeline
from app.core.rate_limiting import RateLimit, rate_limit
from app.core.types import PipelineData, MediaType
from app.core.factories import create_extract_contact_processor, create_text_processor
from app.core.security import get_api_key
from app.services.prediction import PredictionService
from datetime import datetime, timezone

router = APIRouter(prefix="/v1")

@router.get("/health", response_model=HealthResponse)
async def health_check(rate_check=None):
    return HealthResponse(status="healthy")

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
        # Get program manager if available
        program_manager = getattr(request.app.state, "program_manager", None)
        
        # Create service with tracking
        prediction_service = PredictionService(
            request.app.state.model_manager,
            program_manager
        )
        
        # Execute with tracking
        result, execution_info = await prediction_service.predict(query)
        
        # Build metadata including program and model tracking info
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": query.temperature,
            "max_tokens": query.max_tokens
        }
        
        # Add execution info if available
        if execution_info:
            metadata.update({
                "program_id": execution_info.program_id,
                "program_version": execution_info.program_version,
                "program_name": execution_info.program_name,
                "execution_id": execution_info.execution_id,
                "model_id": query.model_id,
                "model_info": execution_info.model_info
            })
        else:
            # Ensure these are always present even without program_manager
            metadata.update({
                "model_id": query.model_id
            })
        
        # Include program and model information in response
        response_data = QueryResponseData(
            response=result,
            model_used=query.model_id,
            metadata=metadata
        )
        
        return QueryResponse(
            success=True, 
            data=response_data,
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline/predict", response_model=PipelineResponse)
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
    program_manager = getattr(request.app.state, "program_manager", None)
    
    # Use factories to create processors based on media type
    if pipeline_req.media_type == MediaType.TEXT:
        processors = [create_text_processor(
            model_manager, 
            pipeline_req.params.get("model_id"),
            program_manager=program_manager,
            metadata={"request_id": str(uuid.uuid4())}
        )]
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
        
        # Extract execution info if available
        response_metadata = {
            "model_id": pipeline_req.params.get("model_id"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if "execution_info" in result.metadata:
            execution_info = result.metadata["execution_info"]
            response_metadata.update({
                "program_id": execution_info.get("program_id"),
                "program_version": execution_info.get("program_version"),
                "program_name": execution_info.get("program_name"),
                "execution_id": execution_info.get("execution_id"),
                "model_info": execution_info.get("model_info", {})
            })
        
        # Include program metadata if available
        if "program_metadata" in result.metadata:
            response_metadata["program_metadata"] = result.metadata["program_metadata"]
        
        response_data = PipelineResponseData(
            content=result.content,
            media_type=result.media_type,
            metadata=result.metadata
        )
        
        return PipelineResponse(
            success=True,
            data=response_data,
            metadata=response_metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/extract-contact", response_model=ExtractContactResponse)
async def process_extract_contact(
    request: Request, 
    pipeline_req: PipelineRequest, 
    api_key: str = Depends(get_api_key),
    rate_check: None = Depends(rate_limit(RateLimit(
        unauthenticated=5,
        authenticated=100,
        window=30
    )))):
    try:
        logging.info(f"Starting contact extraction with model {pipeline_req.params.get('model_id')}")
        model_manager = request.app.state.model_manager
        program_manager = getattr(request.app.state, "program_manager", None)
        
        logging.debug("Creating contact extraction pipeline")
        pipeline = create_extract_contact_processor(
            model_manager,
            pipeline_req.params.get("model_id"),
            program_manager=program_manager,
            metadata={"request_id": str(uuid.uuid4())}
        )
        
        logging.info("Executing pipeline")
        result = await pipeline.execute(PipelineData(
            media_type=MediaType.IMAGE,
            content=pipeline_req.content,
            metadata=pipeline_req.params
        ))
        
        logging.debug("Pipeline execution complete, inspecting history")
        dspy.inspect_history(n=1)
        
        # Extract execution info if available
        execution_info = None
        if "execution_info" in result.metadata:
            execution_info = result.metadata["execution_info"]
        
        # Include program and model information in response metadata
        response_metadata = {
            "model_id": pipeline_req.params.get("model_id"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add execution info if available
        if execution_info:
            response_metadata["program_id"] = execution_info.get("program_id")
            response_metadata["program_version"] = execution_info.get("program_version")
            response_metadata["program_name"] = execution_info.get("program_name")
            response_metadata["execution_id"] = execution_info.get("execution_id")
            response_metadata["model_info"] = execution_info.get("model_info", {})
        
        return ExtractContactResponse(
            success=True,
            data=result.content,
            metadata=response_metadata,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logging.error("Contact extraction failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))