import datetime as dt
import uuid
from typing import Any, Optional

from llm_server.core.types import ProgramMetadata


def ensure_program_metadata_object(metadata: Any) -> Optional[ProgramMetadata]:
    """
    Safely create a ProgramMetadata object from a dictionary,
    ensuring all required fields are present and correctly typed.
    """
    from llm_server.core.types import ProgramMetadata

    if metadata is None:
        return None

    if isinstance(metadata, ProgramMetadata):
        return metadata

    if not isinstance(metadata, dict):
        # Or raise a TypeError, depending on desired strictness
        return None

    # Validate that all REQUIRED fields are present.
    required_fields = {"id", "name", "version", "code_hash"}
    for field in required_fields:
        if field not in metadata or not isinstance(metadata[field], str):
            # This indicates a programming error or bad data.
            # We can't safely create the object.
            return None  # Or raise ValueError(f"Missing or invalid required field: {field}")

    # Build the keyword arguments for the constructor.
    kwargs = {
        "id": metadata["id"],
        "name": metadata["name"],
        "version": metadata["version"],
        "code_hash": metadata["code_hash"],
    }

    # Add OPTIONAL fields if they exist.
    optional_fields = {"description", "tags", "parent_id", "parent_version"}
    for field in optional_fields:
        if field in metadata:
            kwargs[field] = metadata[field]

    # Now it's safe to create the object.
    return ProgramMetadata(**kwargs)


def get_utc_now() -> dt.datetime:
    """Returns the current UTC datetime with timezone information."""
    return dt.datetime.now(dt.timezone.utc)


def format_timestamp(dt=None) -> str:
    """Returns an ISO 8601 formatted timestamp string with timezone information."""
    if dt is None:
        dt = get_utc_now()
    elif dt.tzinfo is None:
        # Convert naive datetime to timezone-aware
        dt = dt.replace(tzinfo=dt.timezone.utc)
    return dt.isoformat()


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
        performance_metrics: Optional[dict[str, Any]] = None,
        model_info: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Collect and structure all metadata for the API response.
        """
        # Start with basic metadata
        metadata: dict[str, Any] = {
            "execution_id": str(uuid.uuid4()),
            "timestamp": format_timestamp(),
        }

        # Use model_info parameter first if provided, then try result metadata
        info = model_info or {}
        if not info and hasattr(result, "metadata") and "model_info" in result.metadata:
            info = result.metadata["model_info"]

        # Add model info in the standardized format
        metadata["model"] = {
            "id": model_id,
            "provider": info.get("provider", "unknown"),
            "base_model": info.get("base_model", model_id),
            "model_name": info.get("model_name", ""),
        }

        # Add program info if available - always convert to object first
        program_metadata = ensure_program_metadata_object(program_metadata)
        if program_metadata:
            metadata["program"] = {
                "id": program_metadata.id,
                "version": program_metadata.version,
                "name": program_metadata.name,
            }

        # Add performance metrics if available - only check the standard location
        if performance_metrics:
            metadata["performance"] = performance_metrics
        elif hasattr(result, "metadata") and "performance" in result.metadata:
            metadata["performance"] = result.metadata["performance"]

        return metadata


def detect_extraction_error(exception):
    """
    Detect the type of extraction error based on the exception.

    Args:
        exception: The exception that was raised

    Returns:
        dict containing error code, message, and technical details
    """
    error_message = str(exception)
    error_info = {
        "code": "EXTRACTION_ERROR",
        "message": "Contact extraction failed",
        "details": error_message,
    }

    # Detect specific error patterns
    if "Images are not yet supported in JSON mode" in error_message:
        error_info.update(
            {
                "code": "UNSUPPORTED_INPUT_FORMAT",
                "message": "Contact extraction failed: image format not supported in fallback processing mode",
            }
        )
    elif "validation error for list" in error_message:
        error_info.update(
            {
                "code": "SCHEMA_VALIDATION_ERROR",
                "message": "Contact extraction failed: model output missing required fields",
            }
        )
    elif "Error parsing field" in error_message:
        error_info.update(
            {
                "code": "OUTPUT_PARSING_ERROR",
                "message": "Contact extraction failed: unable to parse model response",
            }
        )

    return error_info
