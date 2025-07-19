from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel

from llm_server.core.types import ProgramMetadata, ProgramExecutionInfo

# Response schemas
class ProgramListResponse(BaseModel):
    success: bool = True
    data: List[ProgramMetadata]
    timestamp: datetime = datetime.now(timezone.utc)

class ProgramDetailResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any]
    timestamp: datetime = datetime.now(timezone.utc)

class ExecutionHistoryResponse(BaseModel):
    success: bool = True
    data: List[ProgramExecutionInfo]
    timestamp: datetime = datetime.now(timezone.utc)

class ModelInfoResponse(BaseModel):
    success: bool = True
    data: List[Dict[str, Any]]
    timestamp: datetime = datetime.now(timezone.utc)

class EvaluationResponse(BaseModel):
    success: bool = True
    data: List[Dict[str, Any]]
    timestamp: datetime = datetime.now(timezone.utc)

# Define router with relative prefix (no dependencies needed as they come from parent)
programs_router = APIRouter(
    prefix="/programs"  # Changed from /v1/programs to just /programs
)

@programs_router.get("", response_model=ProgramListResponse)
async def list_programs(
    request: Request,
    tags: Optional[str] = None
):
    """List all registered DSPy programs."""
    program_manager = request.app.state.program_manager
    if not program_manager:
        raise HTTPException(status_code=500, detail="Program manager not initialized")
    
    # Parse tags if provided
    tag_list = tags.split(",") if tags else None
    
    programs = program_manager.registry.list_programs(tags=tag_list)
    return ProgramListResponse(data=programs)

@programs_router.get("/{program_id}", response_model=ProgramDetailResponse)
async def get_program_details(
    program_id: str,
    request: Request
):
    """Get details for a specific DSPy program including version history."""
    program_manager = request.app.state.program_manager
    if not program_manager:
        raise HTTPException(status_code=500, detail="Program manager not initialized")
    
    program_tree = program_manager.get_program_tree(program_id)
    if not program_tree:
        raise HTTPException(status_code=404, detail=f"Program {program_id} not found")
    
    return ProgramDetailResponse(data=program_tree)

@programs_router.get("/{program_id}/versions", response_model=ProgramListResponse)
async def list_program_versions(
    program_id: str,
    request: Request
):
    """List all versions of a specific program."""
    program_manager = request.app.state.program_manager
    if not program_manager:
        raise HTTPException(status_code=500, detail="Program manager not initialized")
    
    versions = program_manager.registry.list_program_versions(program_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Program {program_id} not found")
    
    return ProgramListResponse(data=versions)

@programs_router.get("/executions", response_model=ExecutionHistoryResponse)
async def get_execution_history(
    request: Request,
    program_id: Optional[str] = None,
    model_id: Optional[str] = None,
    limit: int = 100
):
    """Get execution history of programs."""
    program_manager = request.app.state.program_manager
    if not program_manager:
        raise HTTPException(status_code=500, detail="Program manager not initialized")
    
    executions = program_manager.get_execution_history(
        program_id=program_id,
        model_id=model_id,
        limit=limit
    )
    
    return ExecutionHistoryResponse(data=executions)

@programs_router.get("/models", response_model=ModelInfoResponse)
async def list_models(
    request: Request
):
    """List all available models from the model config."""
    program_manager = request.app.state.program_manager
    if not program_manager:
        raise HTTPException(status_code=500, detail="Program manager not initialized")
    
    model_info = program_manager.get_available_models()
    return ModelInfoResponse(data=model_info)

@programs_router.get("/{program_id}/evaluations", response_model=EvaluationResponse)
async def get_program_evaluations(
    program_id: str,
    request: Request,
    version: Optional[str] = None,
    model_id: Optional[str] = None
):
    """Get evaluation results for a program."""
    program_manager = request.app.state.program_manager
    if not program_manager:
        raise HTTPException(status_code=500, detail="Program manager not initialized")
    
    evaluations = program_manager.get_evaluation_results(
        program_id=program_id,
        version=version,
        model_id=model_id
    )
    
    return EvaluationResponse(data=evaluations)