import copy
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import ValidationError

from llm_server.api.schemas.requests import PipelineRequest, QueryRequest
from llm_server.api.schemas.responses import (
    ExtractContactResponse,
    HealthResponse,
    PipelineResponse,
    PipelineResponseData,
    QueryResponse,
    QueryResponseData,
)
from llm_server.core import logging
from llm_server.core.config import get_settings
from llm_server.core.metrics_factory import (
    create_metrics_enabled_extract_contact_processor,
    create_metrics_enabled_text_processor,
)
from llm_server.core.rate_limiting import rate_limit
from llm_server.core.security import get_api_key
from llm_server.core.types import MediaType, PipelineData

# --- Helper Functions for Route Handler ---


def _validate_and_parse_request(body: dict[str, Any], request_model: type) -> Any:
    """Validate the incoming request body and parse it with the Pydantic model."""
    if "request" not in body:
        raise HTTPException(status_code=400, detail="Missing 'request' field in body.")
    try:
        return request_model(**body["request"])
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid request: {e}") from e


def _initialize_processor(processor_factory, model_manager, model_id, program_manager):
    """Initialize the processor/pipeline and handle creation errors."""
    try:
        return processor_factory(
            model_manager, model_id, program_manager=program_manager
        )
    except Exception as e:
        logging.error(
            f"Failed to create processor for model {model_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Processor creation failed.") from e


def _prepare_pipeline_data(validated_request: Any) -> PipelineData:
    """Prepare the initial PipelineData object from the validated request."""
    media_type = getattr(validated_request, "media_type", MediaType.TEXT)
    content = getattr(
        validated_request, "content", getattr(validated_request, "prompt", "")
    )
    params = getattr(validated_request, "params", {})

    # Add any extra model parameters to the metadata
    extra_params = getattr(validated_request, "model_extra", {})
    params.update(extra_params)

    return PipelineData(media_type=media_type, content=content, metadata=params)


async def _execute_pipeline(processor: Any, data: PipelineData) -> Any:
    """Execute the pipeline/processor and return the result."""
    if hasattr(processor, "execute"):  # It's a pipeline
        return await processor.execute(data)
    return await processor.process(data)  # It's a single step


def _build_response(
    result: Any, model_id: str, program_manager: Any, response_model: type
):
    """Build the final successful API response object."""
    from llm_server.core.utils import MetadataCollector, ensure_program_metadata_object

    program_metadata = result.metadata.get("program_metadata")
    performance_metrics = result.metadata.get("performance_metrics")
    model_info = program_manager.model_info.get(model_id, {})

    program_metadata = ensure_program_metadata_object(program_metadata)

    response_metadata = MetadataCollector.collect_response_metadata(
        result=result,
        model_id=model_id,
        program_metadata=program_metadata,
        performance_metrics=performance_metrics,
        model_info=model_info,
    )

    # Add any extra parameters from the original request back into the top-level metadata
    if "params" in result.metadata:
        response_metadata.update(result.metadata["params"])

    # Determine the correct response data structure
    if response_model == QueryResponse:
        response_data = QueryResponseData(response=result.content)
    elif response_model == ExtractContactResponse:
        response_data = result.content
    else:  # PipelineResponse
        response_data = PipelineResponseData(
            content=result.content, media_type=result.media_type, metadata={}
        )
        
    final_response = response_model(
        success=True,
        data=response_data,
        metadata=response_metadata,
        timestamp=datetime.now(timezone.utc),
    )

    logging.info(f"Final Response Data: {final_response.model_dump_json(indent=2)}")

    return final_response


# --- Route Handler Factory ---


def create_versioned_route_handler(
    endpoint_name, processor_factory, request_model, response_model
):
    """
    Creates a simplified, low-complexity route handler by delegating tasks
    to helper functions.
    """

    async def route_handler(request: Request, body: dict[str, Any] = Body(...)):  # noqa: B008
        try:
            # 1. Validate and Parse
            validated_request = _validate_and_parse_request(body, request_model)

            # 2. Get Dependencies and Key Info
            model_manager = request.app.state.model_manager
            program_manager = request.app.state.program_manager
            model_id = getattr(
                validated_request, "model_id", None
            ) or validated_request.params.get("model_id")

            if not model_id:
                raise HTTPException(
                    status_code=400, detail="A 'model_id' must be provided."
                )

            logging.info(f"{endpoint_name}: Processing request with model {model_id}")

            # 3. Initialize Processor
            processor = _initialize_processor(
                processor_factory, model_manager, model_id, program_manager
            )

            # 4. Prepare Data and Execute
            pipeline_data = _prepare_pipeline_data(validated_request)
            result = await _execute_pipeline(processor, pipeline_data)

            # 5. Build and Return Success Response
            return _build_response(result, model_id, program_manager, response_model)

        except Exception as e:
            # Centralized error handling for cleaner logic
            logging.error(f"{endpoint_name} failed unexpectedly: {e}", exc_info=True)
            if isinstance(e, HTTPException):
                raise  # Re-raise HTTPException to preserve status code and details
            # For any other exception, return a generic 500 error
            raise HTTPException(
                status_code=500, detail="An internal server error occurred."
            ) from e

    return route_handler


# Create main router with dependencies for all /v1 routes
main_router = APIRouter(
    prefix="/v1",
    dependencies=[
        Depends(get_api_key),  # Apply API key authentication to all routes
        Depends(rate_limit()),  # Apply rate limiting to all routes
    ],
)

# Special case for health check - no auth required
health_router = APIRouter(prefix="/v1")


@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy")


# Route handlers using the factory
@main_router.post("/predict", response_model=QueryResponse)
async def predict(request: Request, body: dict[str, Any] = Body(...)):  # noqa: B008
    handler = create_versioned_route_handler(
        endpoint_name="predict",
        processor_factory=create_metrics_enabled_text_processor,
        request_model=QueryRequest,
        response_model=QueryResponse,
    )
    return await handler(request, body)


@main_router.post("/pipeline/predict", response_model=PipelineResponse)
async def predict_pipeline(request: Request, body: dict[str, Any] = Body(...)):  # noqa: B008
    handler = create_versioned_route_handler(
        endpoint_name="pipeline/predict",
        processor_factory=create_metrics_enabled_text_processor,
        request_model=PipelineRequest,
        response_model=PipelineResponse,
    )
    return await handler(request, body)


@main_router.post("/extract-contact", response_model=ExtractContactResponse)
async def process_extract_contact(request: Request, body: dict[str, Any] = Body(...)):  # noqa: B008
    settings = get_settings()
    try:
        logging.info("--- /v1/extract-contact Request Body ---")
        
        if settings.log_request_bodies:
            # Log the full body if enabled
            logging.info(json.dumps(body, indent=2))
        else:
            # Otherwise, log a sanitized version
            sanitized_body = copy.deepcopy(body)
            if "request" in sanitized_body and "content" in sanitized_body["request"]:
                content = sanitized_body["request"]["content"]
                sanitized_body["request"]["content"] = f"[IMAGE DATA TRUNCATED - Length: {len(str(content))}]"
            logging.info(json.dumps(sanitized_body, indent=2))

        logging.info("--- End Request Body ---")
    except Exception as e:
        logging.error(f"Error logging request body for /v1/extract-contact: {e}")

    handler = create_versioned_route_handler(
        endpoint_name="extract-contact",
        processor_factory=create_metrics_enabled_extract_contact_processor,
        request_model=PipelineRequest,
        response_model=ExtractContactResponse,
    )
    return await handler(request, body)
