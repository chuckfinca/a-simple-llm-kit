from fastapi import APIRouter, Body, Depends, HTTPException, Request

from llm_server.core.factories import create_extract_contact_processor
from llm_server.core.security import get_api_key
from llm_server.core.types import MediaType, PipelineData

main_router = APIRouter(prefix="/v1", dependencies=[Depends(get_api_key)])
health_router = APIRouter(prefix="/v1")

@health_router.get("/health")
async def health_check():
    return {"status": "healthy"}

@main_router.post("/extract-contact")
async def extract_contact(request: Request, body: dict = Body(...)):
    model_manager = request.app.state.model_manager
    
    # Basic validation
    request_data = body.get("request", {})
    image_b64 = request_data.get("content")
    model_id = request_data.get("params", {}).get("model_id", "gpt-4o-mini")

    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing image content.")

    # Create and run the extraction pipeline
    pipeline = create_extract_contact_processor(model_manager, model_id)
    initial_data = PipelineData(media_type=MediaType.IMAGE, content=image_b64)
    result = await pipeline.execute(initial_data)

    # Return the direct result; the backend will wrap it
    return {
        "data": result.content,
        "metadata": result.metadata
    }
    
