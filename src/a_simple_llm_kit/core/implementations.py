import base64
import binascii
import io
from typing import Any

import anyio
import dspy
from PIL import Image

from a_simple_llm_kit.core import logging
from a_simple_llm_kit.core.circuit_breaker import CircuitBreaker
from a_simple_llm_kit.core.protocols import (
    OutputProcessor,
    PipelineStep,
    ProgramMetadata,
)
from a_simple_llm_kit.core.types import (
    ImageProcessingMetadata,
    MediaType,
    PipelineData,
    Usage,
)


class ModelProcessor(PipelineStep):
    """Standard processor for model-based operations with metadata tracking"""

    def __init__(
        self,
        model_manager: Any,
        model_id: str,
        signature_class: type[dspy.Signature],
        input_key: str,
        output_processor: OutputProcessor,
        accepted_types: list[MediaType],
        output_type: MediaType,
        program_metadata: ProgramMetadata | None = None,
    ):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature_class
        self.input_key = input_key
        self.output_processor = output_processor
        self._accepted_types = accepted_types
        self.output_type = output_type
        self.program_metadata = program_metadata

    def _extract_usage_from_history(self, lm: Any, model_id: str) -> Usage:
        """
        Extracts token usage from a model's history, handling provider differences.
        This is the core provider-agnostic logic.
        """
        if not lm or not hasattr(lm, "history") or not lm.history:
            return Usage()

        last_call_usage = lm.history[-1].get("usage", {})

        # Provider-agnostic extraction based on model ID prefix
        if "gpt-" in model_id or "o1-" in model_id:  # OpenAI
            return Usage(
                prompt_tokens=last_call_usage.get("prompt_tokens", 0),
                completion_tokens=last_call_usage.get("completion_tokens", 0),
            )
        elif "claude-" in model_id:  # Anthropic
            return Usage(
                prompt_tokens=last_call_usage.get("input_tokens", 0),
                completion_tokens=last_call_usage.get("output_tokens", 0),
            )
        # Add other providers like gemini here if their format differs

        # Fallback for providers that use the standard 'prompt_tokens' format
        if "prompt_tokens" in last_call_usage:
            return Usage(
                prompt_tokens=last_call_usage.get("prompt_tokens", 0),
                completion_tokens=last_call_usage.get("completion_tokens", 0),
            )

        return Usage()

    @CircuitBreaker()
    async def _protected_predict(self, input_dict: dict[str, Any]) -> Any:
        """
        Internal method that runs the DSPy predictor. This is the operation
        that is protected by the circuit breaker.
        """
        lm = self.model_manager.get_model(self.model_id)
        if not lm:
            raise ValueError(f"Model {self.model_id} not found in ModelManager")

        # Create the predictor instance here, in the main thread.
        predictor = dspy.Predict(self.signature)

        def sync_predictor_call_in_context():
            """
            This wrapper function will run in the worker thread.
            It sets the context and then executes the predictor call.
            """
            # By setting the context here, we guarantee it exists in the
            # same thread that dspy will use to look it up.
            with dspy.context(lm=lm):
                return predictor(**input_dict)

        # Execute our new wrapper function in the thread pool.
        return await anyio.to_thread.run_sync(sync_predictor_call_in_context)  # type: ignore

    async def process(self, data: PipelineData) -> PipelineData:
        """
        This method conforms to the PipelineStep protocol. It is NOT decorated,
        so its signature remains compatible for the type checker.
        """
        # 1. Prepare the input dictionary for the model
        input_dict = {self.input_key: data.content}

        # 2. Call the *protected* internal method
        raw_result = await self._protected_predict(input_dict)

        # --- EXTRACT AND ATTACH USAGE ---
        lm = self.model_manager.get_model(self.model_id)
        usage = self._extract_usage_from_history(lm, self.model_id)
        data.metadata["usage"] = usage  # Attach the usage object to metadata
        logging.info(
            f"Framework extracted token usage: {usage.prompt_tokens} prompt, {usage.completion_tokens} completion"
        )

        # 3. Process the raw output into its final form
        final_result = self.output_processor.process(raw_result, pipeline_data=data)

        # 4. Construct and return the final PipelineData object
        final_metadata = data.metadata.copy()
        final_metadata["processed"] = True

        return PipelineData(
            media_type=self.output_type, content=final_result, metadata=final_metadata
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

    def _apply_orientation(self, image: Image.Image) -> Image.Image:
        """Apply EXIF orientation to the image if necessary."""
        try:
            exif_data = image.getexif()
            if not exif_data:
                return image

            orientation = exif_data.get(0x0112, 1)  # 0x0112 is the EXIF Orientation tag
            logging.info(f"ImageProcessor found EXIF orientation: {orientation}")

            if orientation == 1:  # Normal
                return image
            elif orientation == 2:  # Mirror horizontal
                return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 3:  # Rotate 180
                return image.transpose(Image.Transpose.ROTATE_180)
            elif orientation == 4:  # Mirror vertical
                return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            elif orientation == 5:  # Mirror horizontal and rotate 270 CW
                return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
                    Image.Transpose.ROTATE_270
                )
            elif orientation == 6:  # Rotate 270 CW (or 90 anti-clockwise)
                return image.transpose(Image.Transpose.ROTATE_270)
            elif orientation == 7:  # Mirror horizontal and rotate 90 CW
                return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
                    Image.Transpose.ROTATE_90
                )
            elif orientation == 8:  # Rotate 90 CW
                return image.transpose(Image.Transpose.ROTATE_90)
            return image
        except Exception as e:
            logging.warning(f"Could not apply EXIF orientation: {e}")
            return image  # Return original on error

    @property
    def accepted_media_types(self) -> list[MediaType]:
        return self._accepted_types

    async def process(self, data: PipelineData) -> PipelineData:
        # Wrap content in ImageContent for unified handling
        image_content = ImageContent(data.content)
        pil_image = image_content.pil_image
        corrected_pil_image = self._apply_orientation(pil_image)

        # Get original size before any processing
        original_size = corrected_pil_image.size

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
            processed_pil = corrected_pil_image.resize(
                processed_size, Image.Resampling.LANCZOS
            )
        else:
            # If no resize needed, use original
            processed_pil = corrected_pil_image

        # Convert to dspy.Image before returning
        processed_dspy = dspy.Image.from_PIL(processed_pil)

        processing_metadata = ImageProcessingMetadata(
            mime_type=image_content.detect_mime_type(),
            original_size=original_size,
            processed_size=processed_size,
            compression_ratio=ratio if ratio < 1 else 1.0,
        )

        # Combine existing metadata with the new, structured metadata
        final_metadata = {**data.metadata, "image_processing": processing_metadata}

        return PipelineData(
            media_type=MediaType.IMAGE,
            content=processed_dspy,
            metadata=final_metadata,
        )
