from fastapi import Request, Depends
from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timezone

from llm_server.core.utils import get_utc_now

async def get_versioning_info(
    request: Request,
    model_id: Optional[str] = None,
    program_id: Optional[str] = None,
    program_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Dependency that provides versioning information for API responses.
    It will extract versioning info from the program manager if available.
    
    Args:
        request: The FastAPI request object
        model_id: Optional model ID override
        program_id: Optional program ID override
        program_version: Optional program version override
        
    Returns:
        Dictionary with versioning information
    """
    # Get program manager if available
    program_manager = getattr(request.app.state, "program_manager", None)
    
    # Initialize versioning info with basic fields
    versioning_info = {
        "request_id": str(uuid.uuid4()),
        "timestamp": get_utc_now()
    }
    
    # Create structured model info
    if model_id:
        model_info_dict = {}
        if program_manager and hasattr(program_manager, "model_info"):
            model_info_dict = program_manager.model_info.get(model_id, {})
        
        versioning_info["model"] = {
            "id": model_id,
            **model_info_dict
        }
    
    # Create structured program info
    if program_id:
        program_info = {
            "id": program_id,
            "version": program_version or "unknown"
        }
        
        if program_manager:
            program_metadata = program_manager.registry.get_program_metadata(program_id, program_version)
            if program_metadata:
                program_info["name"] = program_metadata.name
                program_info["version"] = program_metadata.version
        
        versioning_info["program"] = program_info
    
    # If program info wasn't provided, try to get a default program
    elif program_manager and not program_id:
        # For simplicity, find a matching program if possible
        # This is a fallback for endpoints that don't specify a program
        try:
            # Look for a program with matching name based on the route
            path = request.url.path
            endpoint_name = path.split("/")[-1].replace("-", "_")
            
            for prog in program_manager.registry.list_programs():
                if endpoint_name in prog.name.lower():
                    versioning_info["program"] = {
                        "id": prog.id,
                        "version": prog.version,
                        "name": prog.name
                    }
                    break
        except Exception:
            # If anything goes wrong, just continue without the program info
            pass
    
    return versioning_info

class VersionedResponse:
    """
    Base class for creating versioned API responses.
    """
    def __init__(
        self,
        versioning_info: Dict[str, Any] = Depends(get_versioning_info)
    ):
        self.versioning_info = versioning_info
    
    def create_response(
        self, 
        data: Any = None, 
        success: bool = True, 
        error: Optional[str] = None,
        status_code: int = 200,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a versioned API response.
        
        Args:
            data: The response data
            success: Whether the request was successful
            error: Optional error message
            status_code: HTTP status code
            additional_metadata: Any additional metadata to include
            
        Returns:
            Dictionary with the formatted response
        """
        # Combine versioning info with any additional metadata
        metadata = {**self.versioning_info}
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return {
            "success": success,
            "data": data,
            "error": error,
            "metadata": metadata,
            "timestamp": get_utc_now()
        }