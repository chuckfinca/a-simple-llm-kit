import datetime as dt
import uuid
from typing import Any

from a_simple_llm_kit.core.types import (
    ModelInfo,
    ModelResponseInfo,
    PerformanceSummary,
    ProgramMetadata,
    ProgramResponseInfo,
    ResponseMetadata,
)


def ensure_program_metadata_object(metadata: Any) -> ProgramMetadata | None:
    """Safely create a ProgramMetadata object from a dictionary."""
    if metadata is None:
        return None
    if isinstance(metadata, ProgramMetadata):
        return metadata
    if not isinstance(metadata, dict):
        return None
    required_fields = {"id", "name", "version", "code_hash"}
    if not all(field in metadata for field in required_fields):
        return None
    kwargs = {k: metadata[k] for k in ProgramMetadata.model_fields if k in metadata}
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
    """Helper class to enforce consistent metadata collection using Pydantic models."""

    @staticmethod
    def collect_response_metadata(
        model_id: str,
        program_metadata: Any | None = None,
        performance_metrics: dict[str, Any] | None = None,
        model_info: ModelInfo | None = None,
    ) -> dict[str, Any]:
        """Collect and structure all metadata for the API response using"""

        model_data = ModelResponseInfo(
            id=model_id,
            provider=model_info.provider if model_info else "unknown",
            base_model=model_info.base_model if model_info else model_id,
            model_name=model_info.model_name if model_info else model_id,
        )

        program_data = None
        program_meta_obj = ensure_program_metadata_object(program_metadata)
        if program_meta_obj:
            program_data = ProgramResponseInfo(
                id=program_meta_obj.id,
                version=program_meta_obj.version,
                name=program_meta_obj.name,
            )

        execution_id = (
            performance_metrics.get("traceId")
            if performance_metrics
            else str(uuid.uuid4())
        )

        metadata_model = ResponseMetadata(
            execution_id=str(execution_id),
            timestamp=format_timestamp(),
            model=model_data,
            program=program_data,
            performance=PerformanceSummary(**performance_metrics)
            if performance_metrics
            else None,
        )

        return metadata_model.model_dump(by_alias=True, exclude_none=True)


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
