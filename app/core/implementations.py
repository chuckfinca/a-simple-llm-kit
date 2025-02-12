import asyncio
from typing import Any, List, Union, Tuple, Optional
import dspy
from PIL import Image
import io
import base64
from pathlib import Path

from app.core.protocols import PipelineStep, ModelBackend
from app.core.types import MediaType, PipelineData
from app.core.model_interfaces import Signature, ModelOutput
from app.core import logging

class DSPyModelBackend(ModelBackend):
    """DSPy model backend implementation with minimal retry logic and proper async handling"""
    def __init__(
        self,
        model_manager,
        model_id: str,
        signature_class: type[Signature],
    ):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature_class
    
    async def predict(self, input_data: Any) -> ModelOutput:
        max_attempts = 3
        base_delay = 1  # seconds
        
        for attempt in range(max_attempts):
            try:
                lm = self.model_manager.models.get(self.model_id)
                if not lm:
                    raise ValueError(f"Model {self.model_id} not found")
                    
                dspy.configure(lm=lm)
                predictor = dspy.Predict(self.signature)
                
                # Use appropriate input key
                input_key = "image" if self.signature.__name__ == 'ContactExtractor' else "input"
                raw_result = predictor(**{input_key: input_data})
                
                return self.signature.process_output(raw_result)
                    
            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(f"Final attempt failed for model {self.model_id}: {str(e)}")
                    raise
                delay = base_delay * (2 ** attempt)
                logging.warning(
                    f"API call failed: {str(e)}, "
                    f"retrying in {delay} seconds... "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                await asyncio.sleep(delay)

class ModelProcessor:
    """Standard processor for model-based operations"""
    def __init__(self, backend: ModelBackend, accepted_types: list[MediaType], output_type: MediaType):
        self.backend = backend
        self._accepted_types = accepted_types
        self.output_type = output_type
    
    async def process(self, data: PipelineData) -> PipelineData:
        result = await self.backend.predict(data.content)
        return PipelineData(
            media_type=self.output_type,
            content=result,
            metadata={**data.metadata, "processed": True}
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
        """Get image as bytes, converting if necessary"""
        if self._bytes is None:
            if isinstance(self._content, bytes):
                self._bytes = self._content
            elif isinstance(self._content, str):
                if self._content.startswith('data:'):
                    # Handle data URI
                    _, base64_data = self._content.split(',', 1)
                    self._bytes = base64.b64decode(base64_data)
                else:
                    # Handle file path or base64 string
                    try:
                        with open(self._content, 'rb') as f:
                            self._bytes = f.read()
                    except:
                        # Try as base64
                        self._bytes = base64.b64decode(self._content)
        return self._bytes

    @property
    def pil_image(self) -> Image.Image:
        """Get as PIL Image, converting if necessary"""
        if self._pil_image is None:
            self._pil_image = Image.open(io.BytesIO(self.bytes))
            if self._pil_image.mode != 'RGB':
                self._pil_image = self._pil_image.convert('RGB')
        return self._pil_image

    @property
    def data_uri(self) -> str:
        """Get as data URI, converting if necessary"""
        if self._data_uri is None:
            mime_type = self.detect_mime_type()
            base64_data = base64.b64encode(self.bytes).decode('utf-8')
            self._data_uri = f"data:{mime_type};base64,{base64_data}"
        return self._data_uri

    def detect_mime_type(self) -> str:
        """Detect MIME type from image bytes"""
        if self.bytes.startswith(b'\x89PNG\r\n'):
            return 'image/png'
        if self.bytes.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        return 'image/png'  # Default to PNG

class ImageProcessor:
    """Combined image processing step that handles validation, conversion, and preprocessing"""
    def __init__(self, max_size: Tuple[int, int] = (800, 800)):
        self.max_size = max_size
        self._accepted_types = [MediaType.IMAGE]

    @property
    def accepted_media_types(self) -> List[MediaType]:
        return self._accepted_types

    async def process(self, data: PipelineData) -> PipelineData:
        # Wrap content in ImageContent for unified handling
        image = ImageContent(data.content)
        
        # Get original size before any processing
        original_size = image.pil_image.size
        
        # Calculate resize ratio if needed
        ratio = min(self.max_size[0] / original_size[0], 
                    self.max_size[1] / original_size[1])
        
        processed_size = original_size
        if ratio < 1:  # Only resize if image is larger than max_size
            processed_size = (
                int(original_size[0] * ratio),
                int(original_size[1] * ratio)
            )
            # Resize the image
            processed_pil = image.pil_image.resize(
                processed_size, 
                Image.Resampling.LANCZOS
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
                'processed': True,
                'mime_type': image.detect_mime_type(),
                'original_size': original_size,
                'processed_size': processed_size,
                'compression_ratio': ratio if ratio < 1 else 1.0
            }
        )
