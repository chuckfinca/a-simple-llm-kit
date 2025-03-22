import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union

def ensure_program_metadata_object(metadata):
    """
    Ensure program metadata is always a ProgramMetadata object.
    
    Args:
        metadata: Program metadata as dict or object or None
        
    Returns:
        ProgramMetadata object or None
    """
    from app.core.types import ProgramMetadata
    
    if metadata is None:
        return None
        
    if isinstance(metadata, dict):
        # Convert essential fields and ignore extra fields
        required_fields = {"id", "name", "version"}
        clean_dict = {k: metadata.get(k) for k in required_fields if k in metadata}
        
        # Add optional fields if present
        for field in ["code_hash", "description", "tags", "parent_id", "parent_version"]:
            if field in metadata:
                clean_dict[field] = metadata[field]
                
        return ProgramMetadata(**clean_dict)
        
    # Already an object with required attributes
    return metadata

class MetadataCollector:
    """
    Helper class to enforce consistent metadata collection across all processors.
    This ensures all required information is available in a standard format.
    """
    
    @staticmethod
    def collect_response_metadata(
        result, 
        model_id: str, 
        program_metadata: Optional[Any] = None, 
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Collect and structure all metadata for the API response.
        
        Args:
            result: The pipeline or processor result
            model_id: The model ID used
            program_metadata: Optional program metadata (will be converted to object if needed)
            performance_metrics: Optional performance metrics
            
        Returns:
            Dictionary with structured metadata for API response
        """
        # Start with basic metadata
        metadata = {
            "execution_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Try to get model info from result metadata or processor/backend
        model_info = {}
        if hasattr(result, "metadata") and "model_info" in result.metadata:
            model_info = result.metadata["model_info"]
            
        # Add model info in the standardized format
        metadata["model"] = {
            "id": model_id,
            "provider": model_info.get("provider", "unknown"),
            "base_model": model_info.get("base_model", model_id),
            "model_name": model_info.get("model_name", "")
        }
        
        # Add program info if available - always convert to object first
        program_metadata = ensure_program_metadata_object(program_metadata)
        if program_metadata:
            metadata["program"] = {
                "id": program_metadata.id,
                "version": program_metadata.version,
                "name": program_metadata.name
            }
        
        # Add performance metrics if available
        if performance_metrics:
            metadata["performance"] = performance_metrics
        # Also try to extract performance metrics from result
        elif hasattr(result, "metadata") and "performance_metrics" in result.metadata:
            metadata["performance"] = result.metadata["performance_metrics"]
        
        return metadata