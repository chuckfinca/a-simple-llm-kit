from typing import Any

from fastapi import APIRouter, Depends, Request

from llm_server.core.versioning import VersionedResponse, get_versioning_info

# Create a router with a custom prefix
router = APIRouter(prefix="/v1/examples")


@router.post("/simple-example")
async def simple_example(
    request: Request,
    data: dict[str, Any],
    versioning: dict[str, Any] = Depends(get_versioning_info),  # noqa: B008
):
    """
    A simple example endpoint that uses the versioning dependency.
    This approach automatically includes versioning information in the response.
    """
    # Process the request
    result = {"processed": True, "input_size": len(str(data))}

    # Return with versioning info in metadata
    return {
        "success": True,
        "data": result,
        "metadata": versioning,  # Include versioning info here
        "timestamp": versioning["timestamp"],
    }


@router.post("/versioned-response")
async def versioned_response_example(
    request: Request,
    data: dict[str, Any],
    versioned_response: VersionedResponse = Depends(),  # noqa: B008
):
    """
    Example using the VersionedResponse helper.
    This approach provides more flexibility and utility methods.
    """
    # Process the request
    result = {"processed": True, "input_size": len(str(data))}

    # Add any additional metadata specific to this endpoint
    additional_metadata = {"processing_time_ms": 42}

    # Create a properly versioned response
    return versioned_response.create_response(
        data=result, additional_metadata=additional_metadata
    )


@router.post("/custom-model")
async def custom_model_example(
    request: Request,
    data: dict[str, Any],
    model_id: str,
    versioning: dict[str, Any] = Depends(get_versioning_info),  # noqa: B008
):
    """
    Example that specifies a custom model ID.
    """
    # Process the request with the specified model
    result = {"processed": True, "model_used": model_id}

    # Return with versioning info in metadata
    return {
        "success": True,
        "data": result,
        "metadata": versioning,
        "timestamp": versioning["timestamp"],
    }
