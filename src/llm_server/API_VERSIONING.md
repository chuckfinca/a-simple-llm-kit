# API Response Versioning

This document describes the automatic versioning system for API responses. The system ensures that all API responses include program and model versioning information, which is essential for tracking and benchmarking.

## Automatic Enforcement

All API endpoints are automatically checked to ensure they include versioning information through the `VersioningMiddleware`. If an endpoint returns a response without the required versioning fields, the server will return an error.

### Required Fields in Response Metadata

Every API response must include the following fields in its `metadata` section:

- `program_id`: Identifier for the DSPy program used
- `program_version`: Version of the program
- `model_id`: Identifier for the model used

## Adding Versioning to Endpoints

There are three ways to add versioning to your endpoints:

### 1. Using the Versioning Dependency

The simplest approach is to use the `get_versioning_info` dependency:

```python
from fastapi import APIRouter, Depends
from llm_server.core.versioning import get_versioning_info

router = APIRouter()

@router.post("/my-endpoint")
async def my_endpoint(
    data: Dict[str, Any],
    versioning: Dict[str, Any] = Depends(get_versioning_info)
):
    # Process the request
    result = process_data(data)
    
    # Return with versioning info
    return {
        "success": True,
        "data": result,
        "metadata": versioning,  # Include versioning info here
        "timestamp": versioning["timestamp"]
    }
```

### 2. Using the VersionedResponse Helper

For more complex cases, use the `VersionedResponse` helper:

```python
from fastapi import APIRouter, Depends
from llm_server.core.versioning import VersionedResponse

router = APIRouter()

@router.post("/my-endpoint")
async def my_endpoint(
    data: Dict[str, Any],
    versioned_response: VersionedResponse = Depends()
):
    # Process the request
    result = process_data(data)
    
    # Create a properly versioned response
    return versioned_response.create_response(
        data=result,
        additional_metadata={"processing_time_ms": 42}
    )
```

### 3. Specifying Custom Model or Program

If you need to specify a particular model or program:

```python
from fastapi import APIRouter, Depends
from llm_server.core.versioning import get_versioning_info

router = APIRouter()

@router.post("/my-endpoint")
async def my_endpoint(
    data: Dict[str, Any],
    model_id: str,
    versioning: Dict[str, Any] = Depends(
        lambda req: get_versioning_info(
            req, 
            model_id=model_id,
            program_id="my_specific_program",
            program_version="1.0.0"
        )
    )
):
    # Process the request
    result = process_data(data, model_id)
    
    # Return with versioning info
    return {
        "success": True,
        "data": result,
        "metadata": versioning,
        "timestamp": versioning["timestamp"]
    }
```

## Response Format

All versioned responses follow this structure:

```json
{
    "success": true,
    "data": { /* your response data */ },
    "metadata": {
        "program_id": "text_completion",
        "program_version": "1.0.0",
        "program_name": "Predictor",
        "model_id": "gpt-4o-mini", 
        "model_info": {
            "provider": "openai",
            "base_model": "gpt-4o-mini",
            "model_name": "openai/gpt-4o-mini"
        },
        "request_id": "3a7e9f12-d8e2-4b01-9861-4f3a8e72c5a3",
        "timestamp": "2025-03-13T15:42:33.123456Z"
    },
    "timestamp": "2025-03-13T15:42:33.123456Z"
}
```

## Error Response

If an endpoint fails to include the required versioning information, the middleware will return an error:

```json
{
    "success": false,
    "error": "Missing versioning information",
    "details": "Program ID must be included in response metadata; Model ID must be included in response metadata",
    "status_code": 500
}
```

## Excluded Paths

The following paths are excluded from versioning enforcement:

- `/v1/health`
- `/docs`
- `/openapi.json`
- `/redoc`
- `/v1/programs/*`
- `/metrics`

## Implementation Details

The versioning system consists of:

1. **VersioningMiddleware**: Enforces versioning in all API responses
2. **get_versioning_info**: A dependency that provides versioning information
3. **VersionedResponse**: A helper class for creating versioned responses

The middleware intercepts all responses, checks for versioning information, and returns an error if it's missing.