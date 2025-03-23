from typing import Any, Dict, Optional, Union
import uuid
from app.core.implementations import ModelProcessor
from app.core.utils import MetadataCollector
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
from app.core.modules import ExtractContact
from app.core.metrics_factory import create_metrics_enabled_extract_contact_processor, create_metrics_enabled_text_processor



# Create main router with dependencies for all /v1 routes
main_router = APIRouter(
    prefix="/v1",
    dependencies=[
        Depends(get_api_key),  # Apply API key authentication to all routes
        Depends(rate_limit())  # Apply rate limiting to all routes
    ]
)

# Special case for health check - no auth required
health_router = APIRouter(prefix="/v1")

@health_router.get("/health", response_model=HealthResponse)
async def health_check():
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
        body: Dict[str, Any] = Body(...)
    ):
        try:
            if "request" not in body:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request format. Expected 'request' field in request body."
                )
                
            req_data = body["request"]
                
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
            
            # Extract program metadata - we need this for versioning
            program_metadata = None
            performance_metrics = None
            
            # Look for program metadata directly in result
            if hasattr(result, "metadata"):
                # Check for program_metadata in result
                if "program_metadata" in result.metadata:
                    program_metadata = result.metadata["program_metadata"]
                
                # Check for performance_metrics
                if "performance_metrics" in result.metadata:
                    performance_metrics = result.metadata["performance_metrics"]
            
            # If no program metadata found, search common locations
            if program_metadata is None:
                # First try looking for it in the processor's backend if available
                if hasattr(processor, "backend") and hasattr(processor.backend, "program_metadata"):
                    program_metadata = processor.backend.program_metadata
                
                # For metrics-enabled processors, check the pipeline object
                elif hasattr(processor, "pipeline") and hasattr(processor.pipeline, "steps"):
                    for step in processor.pipeline.steps:
                        if hasattr(step, "backend") and hasattr(step.backend, "program_metadata"):
                            program_metadata = step.backend.program_metadata
                            break
                        
            # Import at this point to avoid circular imports
            from app.core.utils import MetadataCollector, ensure_program_metadata_object
            
            # Ensure program_metadata is a proper object before passing to collector
            program_metadata = ensure_program_metadata_object(program_metadata)
            
            # Get model_info directly from program_manager
            model_info = {}
            if program_manager and hasattr(program_manager, 'model_info') and model_id in program_manager.model_info:
                model_info = program_manager.model_info.get(model_id, {})
                logging.info(f"Using model info from program_manager: {model_info}")

            # Collect all metadata in a clean, structured format
            response_metadata = MetadataCollector.collect_response_metadata(
                result=result,
                model_id=model_id,
                program_metadata=program_metadata,
                performance_metrics=performance_metrics,
                model_info=model_info
            )
            
            # Add any additional parameters to metadata
            if hasattr(validated_request, "model_extra") and validated_request.model_extra:
                for key, value in validated_request.model_extra.items():
                    # Add these at the top level of metadata
                    response_metadata[key] = value
            
            # Extract the actual response content
            response_content = result.content
            if hasattr(result.content, 'output'):
                response_content = result.content.output
            
            # Create response based on the model type
            if response_model == QueryResponse:
                return QueryResponse(
                    success=True,
                    data=QueryResponseData(
                        response=response_content
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
                        metadata={}  # No nested metadata here
                    ),
                    metadata=response_metadata,
                    timestamp=datetime.now(timezone.utc)
                )
            elif response_model == ExtractContactResponse:
                # Handle the special case for ExtractContact
                contact_data = result.content
                if hasattr(result.content, 'output'):
                    contact_data = result.content.output
                
                # Make sure we have an ExtractContact instance
                if not isinstance(contact_data, ExtractContact):
                    # Try to convert from dict if needed
                    if isinstance(contact_data, dict):
                        contact_data = ExtractContact(**contact_data)
                    else:
                        logging.error(f"{endpoint_name}: Cannot convert result to ExtractContact: {type(contact_data)}")
                        raise HTTPException(
                            status_code=500,
                            detail="Invalid contact data format returned from processor"
                        )
                
                return ExtractContactResponse(
                    success=True,
                    data=contact_data,
                    metadata=response_metadata,
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logging.error(f"{endpoint_name} failed", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # Return the route handler function
    return route_handler

# Route handlers using the factory
@main_router.post("/predict", response_model=QueryResponse)
async def predict(request: Request, body: Dict[str, Any] = Body(...)):
    handler = create_versioned_route_handler(
        endpoint_name="predict",
        processor_factory=create_metrics_enabled_text_processor,
        request_model=QueryRequest,
        response_model=QueryResponse
    )
    return await handler(request, body)

@main_router.post("/pipeline/predict", response_model=PipelineResponse)
async def predict_pipeline(request: Request, body: Dict[str, Any] = Body(...)):
    handler = create_versioned_route_handler(
        endpoint_name="pipeline/predict",
        processor_factory=create_metrics_enabled_text_processor,
        request_model=PipelineRequest,
        response_model=PipelineResponse
    )
    return await handler(request, body)

@main_router.post("/extract-contact", response_model=ExtractContactResponse)
async def process_extract_contact(request: Request, body: Dict[str, Any] = Body(...)):
    handler = create_versioned_route_handler(
        endpoint_name="extract-contact",
        processor_factory=create_metrics_enabled_extract_contact_processor,
        request_model=PipelineRequest,
        response_model=ExtractContactResponse
    )
    return await handler(request, body)