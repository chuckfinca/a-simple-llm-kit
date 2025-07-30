import asyncio
import base64
import binascii
import io
import uuid
from typing import Any, Optional, Union

import dspy
from PIL import Image

from llm_server.core import logging
from llm_server.core.model_interfaces import ModelOutput, Signature
from llm_server.core.protocols import ModelBackend
from llm_server.core.types import MediaType, PipelineData


class DSPyModelBackend(ModelBackend):
    """DSPy model backend implementation."""

    def __init__(
        self,
        model_manager,
        model_id: str,
        signature_class: type[Signature],
        program_manager=None, # This is now ignored but kept for compatibility
    ):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature_class
        # The program_manager is no longer used in the simplified server
        self.program_manager = None
        self.program_metadata = None

        # Fulfill the ModelBackend protocol by adding these properties.
        # They are not used in the simplified server but are required by the interface.
        self.last_prompt_tokens: Optional[int] = None
        self.last_completion_tokens: Optional[int] = None

    def _determine_input_key(
        self, signature_class: type[Signature], input_data: Any
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

        # If no input fields found or only one input field, use generic approach
        if len(input_fields) <= 1:
            # Common input types
            if signature_class.__name__ == "ContactExtractor":
                return {"image": input_data}
            else:
                return {"input": input_data}

        # For complex inputs with multiple fields, we need content to be a dict already
        if isinstance(input_data, dict):
            return input_data
        else:
            logging.warning(
                f"Complex signature {signature_class.__name__} with multiple inputs "
                f"but received non-dict input. Using 'input' as default key."
            )
            return {"input": input_data}

    async def predict(self, input: Any) -> ModelOutput:
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
                
                # We now call the new robust processor from here.
                # This ensures the dual-path logic is always used.
                from llm_server.core.output_processors import ContactExtractorProcessor
                processor = ContactExtractorProcessor()
                return processor.process(raw_result)
                # --- END OF THE SIMPLIFIED LOGIC ---
    
            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(f"Final attempt failed for model {self.model_id}: {str(e)}", extra={"trace_id": trace_id})
                    raise
                delay = base_delay * (2**attempt)
                logging.warning(f"API call failed: {str(e)}, retrying in {delay} seconds... (attempt {attempt + 1}/{max_attempts})", extra={"trace_id": trace_id})
                await asyncio.sleep(delay)
    
        raise RuntimeError("The prediction loop completed without returning or raising an error.")

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
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.backend = backend
        self._accepted_types = accepted_types
        self.output_type = output_type
        self.metadata = metadata or {}

    async def process(self, data: PipelineData) -> PipelineData:
        result = await self.backend.predict(data.content)

        # Combine metadata from multiple sources with priority:
        # 1. Model execution specific metadata (from result)
        # 2. Processor instance metadata
        # 3. Input data metadata
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

    def __init__(self, content: Union[str, bytes]):
        self._content = content
        self._bytes: Optional[bytes] = None
        self._pil_image: Optional[Image.Image] = None
        self._data_uri: Optional[str] = None

    @property
    def bytes(self) -> bytes:
        """Get image as bytes, converting if necessary and fixing padding errors."""
        import base64

        from llm_server.core import logging  # Local import for clarity
    
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
    
                # --- Robust Decoding Logic with Enhanced Logging ---
                try:
                    # 1. Log the initial state for diagnostics
                    initial_len = len(content_str)
                    logging.info(f"ImageContent: Attempting to decode base64 string. Initial length: {initial_len}")
                    logging.debug(f"ImageContent: First 30 chars of string: '{content_str[:30]}...'")
                    logging.debug(f"ImageContent: Last 30 chars of string: '...{content_str[-30:]}'")
    
                    # 2. Calculate and log required padding
                    missing_padding = len(content_str) % 4
                    logging.debug(f"ImageContent: Calculated missing padding characters: {missing_padding}")
    
                    if missing_padding:
                        padding_to_add = '=' * (4 - missing_padding)
                        content_str += padding_to_add
                        logging.info(f"ImageContent: Applied '{padding_to_add}' padding. New length: {len(content_str)}")
                    else:
                        logging.info("ImageContent: String length is valid, no padding needed.")
    
                    # 3. Attempt the decode operation
                    self._bytes = base64.b64decode(content_str)
                    logging.info(f"ImageContent: Successfully decoded base64 string to {len(self._bytes)} bytes.")
    
                except (binascii.Error, TypeError) as e:
                    # This log is crucial if the error persists
                    logging.error(f"ImageContent: base64.b64decode FAILED even after padding attempt. Final length was {len(content_str)}.", exc_info=True)
                    raise ValueError(f"Content is not a valid base64 string: {e}") from e
                # --- End of Enhanced Logic ---
    
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
            content=processed_dspy,  # Now returning dspy.Image instead of PIL Image
            metadata={
                **data.metadata,
                "processed": True,
                "mime_type": image.detect_mime_type(),
                "original_size": original_size,
                "processed_size": processed_size,
                "compression_ratio": ratio if ratio < 1 else 1.0,
            },
        )
