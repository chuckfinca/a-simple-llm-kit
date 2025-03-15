from typing import Any, Dict, Optional, Union
import uuid
import dspy
from fastapi import APIRouter, Depends, HTTPException, Request, Body
from app.api.schemas.requests import QueryRequest, PipelineRequest, RequestWrapper
from app.api.schemas.responses import ExtractContactResponse, HealthResponse, PipelineResponse, PipelineResponseData, QueryResponse, QueryResponseData
from app.core import logging
from app.core.pipeline import Pipeline
from app.core.rate_limiting import RateLimit, rate_limit
from app.core.types import PipelineData, MediaType
from app.core.factories import create_extract_contact_processor, create_text_processor
from app.core.security import get_api_key
from app.core.versioning import get_versioning_info
from app.services.prediction import PredictionService
from datetime import datetime, timezone
from pydantic import ValidationError

router = APIRouter(prefix="/v1")

@router.get("/health", response_model=HealthResponse)
async def health_check(rate_check=None):
    return HealthResponse(status="healthy")

@router.post("/predict", response_model=QueryResponse)
async def predict(
    request: Request, 
    body: Dict[str, Any] = Body(...),  # Accept any JSON body
    api_key: str = Depends(get_api_key),
    rate_check: None = Depends(rate_limit(RateLimit(
        unauthenticated=5,
        authenticated=100,
        window=30
    )))):
    try:
        # Support both old and new formats during transition
        req_data = None
        
        # Check for new format with "request" field
        if "request" in body:
            req_data = body["request"]
        # Backward compatibility with old {"query": {"req": {...}}} format
        elif "query" in body and "req" in body["query"]:
            req_data = body["query"]["req"]
            logging.warning("Deprecated request format detected. Please update to use {'request': {...}} format")
        
        if not req_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid request format. Expected 'request' field in request body."
            )
            
        # Now create a proper QueryRequest object, allowing extra fields
        try:
            query_request = QueryRequest(**req_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request parameters: {str(e)}"
            )
        
        # Get program manager if available
        program_manager = getattr(request.app.state, "program_manager", None)
        
        # Create service with tracking
        prediction_service = PredictionService(
            request.app.state.model_manager,
            program_manager
        )
        
        # Execute with tracking
        result, execution_info = await prediction_service.predict(query_request)
        
        # Get versioning info
        versioning = await get_versioning_info(
            request,
            model_id=query_request.model_id
        )
        
        # Build metadata including program and model tracking info
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": query_request.temperature,
            "max_tokens": query_request.max_tokens
        }
        
        # Add any additional parameters to metadata
        if hasattr(query_request, "model_extra") and query_request.model_extra:
            for key, value in query_request.model_extra.items():
                if key not in metadata and key not in ["prompt", "model_id"]:
                    metadata[key] = value
        
        # Add execution info if available
        if execution_info:
            metadata.update({
                "program_id": execution_info.program_id,
                "program_version": execution_info.program_version,
                "program_name": execution_info.program_name,
                "execution_id": execution_info.execution_id,
                "model_id": query_request.model_id,
                "model_info": execution_info.model_info
            })
        else:
            # Ensure these are always present even without program_manager
            metadata.update({
                "model_id": query_request.model_id
            })
        
        # Include program and model information in response
        response_data = QueryResponseData(
            response=result,
            model_used=query_request.model_id,
            metadata=metadata
        )
        
        return QueryResponse(
            success=True, 
            data=response_data,
            metadata=metadata
        )
    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline/predict", response_model=PipelineResponse)
async def predict_pipeline(
    request: Request, 
    body: Dict[str, Any] = Body(...),  # Accept any JSON body
    api_key: str = Depends(get_api_key),
    rate_check: None = Depends(rate_limit(RateLimit(
        unauthenticated=5,
        authenticated=100,
        window=30
    )))):
    try:
        # Support both old and new formats during transition
        req_data = None
        
        # Check for new format with "request" field
        if "request" in body:
            req_data = body["request"]
        # Backward compatibility with old {"query": {"req": {...}}} format
        elif "query" in body and "req" in body["query"]:
            req_data = body["query"]["req"]
            logging.warning("Deprecated request format detected. Please update to use {'request': {...}} format")
        
        if not req_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid request format. Expected 'request' field in request body."
            )
            
        # Create a proper PipelineRequest object
        try:
            pipeline_req = PipelineRequest(**req_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request parameters: {str(e)}"
            )
        
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
        
        # Add any additional parameters to metadata
        if hasattr(pipeline_req, "model_extra") and pipeline_req.model_extra:
            for key, value in pipeline_req.model_extra.items():
                if key not in response_metadata:
                    response_metadata[key] = value
        
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
        logging.error(f"Error in pipeline/predict endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/extract-contact", response_model=ExtractContactResponse)
async def process_extract_contact(
    request: Request, 
    body: Dict[str, Any] = Body(...),  # Accept any JSON body
    api_key: str = Depends(get_api_key),
    rate_check: None = Depends(rate_limit(RateLimit(
        unauthenticated=5,
        authenticated=100,
        window=30
    )))):
    try:
        # Support both old and new formats during transition
        req_data = None
        
        # Check for new format with "request" field
        if "request" in body:
            req_data = body["request"]
        # Backward compatibility with old {"query": {"req": {...}}} format
        elif "query" in body and "req" in body["query"]:
            req_data = body["query"]["req"]
            logging.warning("Deprecated request format detected. Please update to use {'request': {...}} format")
        
        if not req_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid request format. Expected 'request' field in request body."
            )
            
        # Create a proper PipelineRequest object
        try:
            pipeline_req = PipelineRequest(**req_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request parameters: {str(e)}"
            )
        
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
        
        # Add any additional parameters to metadata
        if hasattr(pipeline_req, "model_extra") and pipeline_req.model_extra:
            for key, value in pipeline_req.model_extra.items():
                if key not in response_metadata:
                    response_metadata[key] = value
        
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