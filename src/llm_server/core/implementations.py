import asyncio
import base64
import binascii
import io
import uuid
from typing import Any

import dspy
from PIL import Image

from llm_server.core import logging
from llm_server.core.circuit_breaker import CircuitBreaker
from llm_server.core.model_interfaces import ModelOutput
from llm_server.core.output_processors import DefaultOutputProcessor
from llm_server.core.protocols import ModelBackend, OutputProcessor
from llm_server.core.types import MediaType, PipelineData


class DSPyModelBackend(ModelBackend):
    """DSPy model backend implementation."""

    def __init__(
        self,
        model_manager,
        model_id: str,
        signature_class: type[dspy.Signature],
        output_processor: OutputProcessor | None = None,
    ):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature_class

        self.output_processor = output_processor or DefaultOutputProcessor()

        # Fulfill the ModelBackend protocol by adding these properties.
        # They are not used in the simplified server but are required by the interface.
        self.last_prompt_tokens: int | None = None
        self.last_completion_tokens: int | None = None

    def _determine_input_key(
        self, signature_class: type[dspy.Signature], input_data: Any
    ) -> dict[str, Any]:
        """
        Determine the appropriate input key based on the signature class and input data.
        This handles different input formats for different program types.
        """
        # Get signature fields to determine proper input key
        input_fields = {}

        # Analyze input fields from the signature class
        annotations = getattr(signature_class, "__annotations__", {})
        for name, field in annotations.items():
            # Look for fields marked as dspy.InputField
            if hasattr(field, "__origin__") and field.__origin__ == dspy.InputField:
                input_fields[name] = field

        # If there's only one input field, use its name as the key.
        # This is robust and doesn't rely on hardcoded class names.
        if len(input_fields) == 1:
            key = list(input_fields.keys())[0]
            return {key: input_data}

        # For complex signatures, the input must be a dict.
        if isinstance(input_data, dict):
            return input_data

        # Fallback if a non-dict is passed for a multi-input signature
        raise TypeError(
            f"Signature {signature_class.__name__} has multiple inputs, "
            "but a non-dict input was provided."
        )

    @CircuitBreaker()
    async def predict(self, input: Any) -> ModelOutput:  # type: ignore[override]
        max_attempts = 3
        base_delay = 1  # seconds
        trace_id = str(uuid.uuid4())
        input_dict = self._determine_input_key(self.signature, input)

        for attempt in range(max_attempts):
            try:
                lm = self.model_manager.models.get(self.model_id)
                if not lm:
                    raise ValueError(f"Model {self.model_id} not found")

                logging.warning(
                    f"Executing {self.signature.__name__} without program tracking. "
                    "This is the expected behavior for the simplified llm-server."
                )

                dspy.configure(lm=lm)
                predictor = dspy.Predict(self.signature)

                raw_result = predictor(**input_dict)

                return self.output_processor.process(raw_result)

            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(
                        f"Final attempt failed for model {self.model_id}: {str(e)}",
                        extra={"trace_id": trace_id},
                    )
                    raise
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"API call failed: {str(e)}, retrying in {delay} seconds... (attempt {attempt + 1}/{max_attempts})",
                    extra={"trace_id": trace_id},
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            "The prediction loop completed without returning or raising an error."
        )

    def get_lm_history(self):
        """Safely get LM history without exposing the full LM object"""
        try:
            lm = self.model_manager.models.get(self.model_id)
            if hasattr(lm, "history"):
                return lm.history
        except Exception as e:
            logging.warning(f"Could not access LM history: {str(e)}")
        return []


class ModelProcessor:
    """Standard processor for model-based operations with metadata tracking"""

    def __init__(
        self,
        backend: ModelBackend,
        accepted_types: list[MediaType],
        output_type: MediaType,
        metadata: dict[str, Any] | None = None,
    ):
        self.backend = backend
        self._accepted_types = accepted_types
        self.output_type = output_type
        self.metadata = metadata or {}

    async def process(self, data: PipelineData) -> PipelineData:
        result = await self.backend.predict(data.content)

        # Combine metadata from multiple sources with priority:
        combined_metadata = {**data.metadata, **self.metadata}

        # Add execution info if available from the result
        result_metadata = {}
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            result_metadata = result.metadata

        # Get program_metadata from backend in a consistent format
        if hasattr(self.backend, "program_metadata"):
            from llm_server.core.utils import ensure_program_metadata_object

            program_metadata = ensure_program_metadata_object(
                self.backend.program_metadata
            )
            if program_metadata:
                combined_metadata["program_metadata"] = program_metadata

        # Update with result metadata
        combined_metadata.update(result_metadata)
        combined_metadata["processed"] = True

        return PipelineData(
            media_type=self.output_type, content=result, metadata=combined_metadata
        )

    @property
    def accepted_media_types(self) -> list[MediaType]:
        return self._accepted_types


class ImageContent:
    """Wrapper class to handle different image formats and conversions"""

    def __init__(self, content: str | bytes):
        self._content = content
        self._bytes: bytes | None = None
        self._pil_image: Image.Image | None = None
        self._data_uri: str | None = None

    @property
    def bytes(self) -> bytes:
        """Get image as bytes, converting if necessary and fixing padding errors."""
        import base64

        if self._bytes is None:
            if isinstance(self._content, bytes):
                self._bytes = self._content
            elif isinstance(self._content, str):
                content_str = self._content

                if content_str.startswith("data:"):
                    try:
                        _, base64_data = content_str.split(",", 1)
                        content_str = base64_data
                    except ValueError as e:
                        raise ValueError(f"Invalid data URI provided: {e}") from e

                try:
                    # Log the initial state for diagnostics
                    initial_len = len(content_str)
                    logging.info(
                        f"ImageContent: Attempting to decode base64 string. Initial length: {initial_len}"
                    )
                    logging.debug(
                        f"ImageContent: First 30 chars of string: '{content_str[:30]}...'"
                    )
                    logging.debug(
                        f"ImageContent: Last 30 chars of string: '...{content_str[-30:]}'"
                    )

                    # Calculate and log required padding
                    missing_padding = len(content_str) % 4
                    logging.debug(
                        f"ImageContent: Calculated missing padding characters: {missing_padding}"
                    )

                    if missing_padding:
                        padding_to_add = "=" * (4 - missing_padding)
                        content_str += padding_to_add
                        logging.info(
                            f"ImageContent: Applied '{padding_to_add}' padding. New length: {len(content_str)}"
                        )
                    else:
                        logging.info(
                            "ImageContent: String length is valid, no padding needed."
                        )

                    # Attempt the decode operation
                    self._bytes = base64.b64decode(content_str)
                    logging.info(
                        f"ImageContent: Successfully decoded base64 string to {len(self._bytes)} bytes."
                    )

                except (binascii.Error, TypeError) as e:
                    logging.error(
                        f"ImageContent: base64.b64decode FAILED even after padding attempt. Final length was {len(content_str)}.",
                        exc_info=True,
                    )
                    raise ValueError(
                        f"Content is not a valid base64 string: {e}"
                    ) from e

        if self._bytes is None:
            raise TypeError("Image content could not be converted to bytes.")

        return self._bytes

    @property
    def pil_image(self) -> Image.Image:
        """Get as PIL Image, converting if necessary"""
        if self._pil_image is None:
            self._pil_image = Image.open(io.BytesIO(self.bytes))
            if self._pil_image.mode != "RGB":
                self._pil_image = self._pil_image.convert("RGB")
        return self._pil_image

    @property
    def data_uri(self) -> str:
        """Get as data URI, converting if necessary"""
        if self._data_uri is None:
            mime_type = self.detect_mime_type()
            base64_data = base64.b64encode(self.bytes).decode("utf-8")
            self._data_uri = f"data:{mime_type};base64,{base64_data}"
        return self._data_uri

    def detect_mime_type(self) -> str:
        """Detect MIME type from image bytes"""
        if self.bytes.startswith(b"\x89PNG\r\n"):
            return "image/png"
        if self.bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        return "image/png"  # Default to PNG


class ImageProcessor:
    """Combined image processing step that handles validation, conversion, and preprocessing"""

    def __init__(self, max_size: tuple[int, int] = (800, 800)):
        self.max_size = max_size
        self._accepted_types = [MediaType.IMAGE]

    @property
    def accepted_media_types(self) -> list[MediaType]:
        return self._accepted_types

    async def process(self, data: PipelineData) -> PipelineData:
        # Wrap content in ImageContent for unified handling
        image = ImageContent(data.content)

        # Get original size before any processing
        original_size = image.pil_image.size

        # Calculate resize ratio if needed
        ratio = min(
            self.max_size[0] / original_size[0], self.max_size[1] / original_size[1]
        )

        processed_size = original_size
        if ratio < 1:  # Only resize if image is larger than max_size
            processed_size = (
                int(original_size[0] * ratio),
                int(original_size[1] * ratio),
            )
            # Resize the image
            processed_pil = image.pil_image.resize(
                processed_size, Image.Resampling.LANCZOS
            )
        else:
            # If no resize needed, use original
            processed_pil = image.pil_image

        # Convert to dspy.Image before returning
        processed_dspy = dspy.Image.from_PIL(processed_pil)

        return PipelineData(
            media_type=MediaType.IMAGE,
            content=processed_dspy,
            metadata={
                **data.metadata,
                "processed": True,
                "mime_type": image.detect_mime_type(),
                "original_size": original_size,
                "processed_size": processed_size,
                "compression_ratio": ratio if ratio < 1 else 1.0,
            },
        )
