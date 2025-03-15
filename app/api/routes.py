from typing import Any, Dict, Optional, Union
import uuid
import dspy
from fastapi import APIRouter, Depends, HTTPException, Request, Body
from app.api.schemas.requests import QueryRequest, PipelineRequest
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
from pydantic import ValidationError, BaseModel

router = APIRouter(prefix="/v1")

@router.get("/health", response_model=HealthResponse)
async def health_check(rate_check=None):
    return HealthResponse(status="healthy")

def create_versioned_route_handler(endpoint_name, processor_factory, request_model, response_model):
    """
    Creates a route handler with proper versioning validation.
    
    Args:
        endpoint_name: Name of the endpoint for logging
        processor_factory: Function that creates the processor/pipeline
        request_model: Pydantic model for request validation
        response_model: Response model class for the response
        
    Returns:
        A route handler function
    """
    async def route_handler(
        request: Request, 
        body: Dict[str, Any] = Body(...),
        api_key: str = Depends(get_api_key),
        rate_check: None = Depends(rate_limit(RateLimit(
            unauthenticated=5,
            authenticated=100,
            window=30
        )))
    ):
        try:
            # Support both old and new formats during transition
            req_data = None
            
            # Check for new format with "request" field
            if "request" in body:
                req_data = body["request"]
            # Backward compatibility with old {"query": {"req": {...}}} format
            elif "query" in body and "req" in body["query"]:
                req_data = body["query"]["req"]
                logging.warning(f"{endpoint_name}: Deprecated request format detected. Please update to use {'request': {...}} format")
            
            if not req_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request format. Expected 'request' field in request body."
                )
                
            # Create a proper request object
            try:
                validated_request = request_model(**req_data)
            except ValidationError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid request parameters: {str(e)}"
                )
            
            # Get dependencies
            model_manager = request.app.state.model_manager
            program_manager = getattr(request.app.state, "program_manager", None)
            
            if not program_manager:
                raise HTTPException(
                    status_code=500,
                    detail="Program manager is not initialized. This is required for versioning."
                )
            
            # Get request-specific fields
            model_id = None
            if hasattr(validated_request, "params") and isinstance(validated_request.params, dict):
                model_id = validated_request.params.get("model_id")
            else:
                model_id = getattr(validated_request, "model_id", None)
                
            if not model_id:
                raise HTTPException(
                    status_code=400,
                    detail="A 'model_id' must be provided in the request"
                )
            
            logging.info(f"{endpoint_name}: Processing request with model {model_id}")
            
            try:
                # Create processor or pipeline
                processor = processor_factory(
                    model_manager,
                    model_id,
                    program_manager=program_manager,
                    metadata={"request_id": str(uuid.uuid4())}
                )
            except ValueError as e:
                # Specific error for program registration issues
                logging.error(f"{endpoint_name}: Failed to create processor: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create processor: {str(e)}"
                )
            except Exception as e:
                logging.error(f"{endpoint_name}: Failed to create processor: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create processor: {str(e)}"
                )
            
            # Prepare pipeline data
            media_type = getattr(validated_request, "media_type", MediaType.TEXT)
            content = getattr(validated_request, "content", 
                             getattr(validated_request, "prompt", ""))
            metadata = getattr(validated_request, "params", {})
            
            # For non-pipeline requests, add standard fields to metadata
            if not hasattr(validated_request, "params"):
                metadata = {}
                if hasattr(validated_request, "temperature"):
                    metadata["temperature"] = validated_request.temperature
                if hasattr(validated_request, "max_tokens"):
                    metadata["max_tokens"] = validated_request.max_tokens
                if hasattr(validated_request, "model_id"):
                    metadata["model_id"] = validated_request.model_id
            
            # Execute pipeline or processor
            try:
                # Check if it's a pipeline or single processor
                if hasattr(processor, "execute"):  # It's a pipeline
                    result = await processor.execute(PipelineData(
                        media_type=media_type,
                        content=content,
                        metadata=metadata
                    ))
                else:  # It's a single processor
                    result = await processor.process(PipelineData(
                        media_type=media_type,
                        content=content,
                        metadata=metadata
                    ))
            except Exception as e:
                logging.error(f"{endpoint_name}: Execution failed: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Execution failed: {str(e)}"
                )
            
            # Extract execution info
            execution_info = {}
            if isinstance(result.metadata, dict) and "execution_info" in result.metadata:
                execution_info = result.metadata["execution_info"]
            else:
                logging.error(f"{endpoint_name}: Execution info missing from result metadata")
                raise HTTPException(
                    status_code=500,
                    detail="Execution info missing from result. This is required for versioning."
                )
            
            # Build response metadata
            response_metadata = {
                "model_id": model_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add any additional parameters to metadata
            if hasattr(validated_request, "model_extra") and validated_request.model_extra:
                for key, value in validated_request.model_extra.items():
                    if key not in response_metadata:
                        response_metadata[key] = value
            
            # Add execution info - required for versioning
            if execution_info:
                for key in ["program_id", "program_version", "program_name", "execution_id", "model_info"]:
                    if key in execution_info:
                        response_metadata[key] = execution_info.get(key)
            
            # Verify versioning info is present before returning
            required_fields = ["program_id", "program_version", "model_id"]
            missing_fields = [field for field in required_fields if field not in response_metadata]
            
            if missing_fields:
                missing_str = ", ".join(missing_fields)
                logging.error(f"{endpoint_name}: Missing required versioning fields: {missing_str}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required versioning fields: {missing_str}. Check program registration."
                )
            
            # Create response with the appropriate model
            if response_model == QueryResponse:
                return QueryResponse(
                    success=True,
                    data=QueryResponseData(
                        response=result.content,
                        model_used=model_id,
                        metadata=response_metadata
                    ),
                    metadata=response_metadata,
                    timestamp=datetime.now(timezone.utc)
                )
            elif response_model == PipelineResponse:
                return PipelineResponse(
                    success=True,
                    data=PipelineResponseData(
                        content=result.content,
                        media_type=result.media_type,
                        metadata=result.metadata
                    ),
                    metadata=response_metadata,
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                return response_model(
                    success=True,
                    data=result.content,
                    metadata=response_metadata,
                    timestamp=datetime.now(timezone.utc)
                )
            
        except HTTPException:
            # Re-raise HTTP exceptions without logging (they're already handled)
            raise
        except Exception as e:
            logging.error(f"{endpoint_name} failed", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return route_handler

# Updated route handlers using the factory

@router.post("/predict", response_model=QueryResponse)
async def predict(request: Request, body: Dict[str, Any] = Body(...), **kwargs):
    handler = create_versioned_route_handler(
        endpoint_name="predict",
        processor_factory=create_text_processor,
        request_model=QueryRequest,
        response_model=QueryResponse
    )
    return await handler(request, body, **kwargs)

@router.post("/pipeline/predict", response_model=PipelineResponse)
async def predict_pipeline(request: Request, body: Dict[str, Any] = Body(...), **kwargs):
    handler = create_versioned_route_handler(
        endpoint_name="pipeline/predict",
        processor_factory=create_text_processor,  # You might need to create a different factory
        request_model=PipelineRequest,
        response_model=PipelineResponse
    )
    return await handler(request, body, **kwargs)

@router.post("/extract-contact", response_model=ExtractContactResponse)
async def process_extract_contact(request: Request, body: Dict[str, Any] = Body(...), **kwargs):
    handler = create_versioned_route_handler(
        endpoint_name="extract-contact",
        processor_factory=create_extract_contact_processor,
        request_model=PipelineRequest,
        response_model=ExtractContactResponse
    )
    return await handler(request, body, **kwargs)