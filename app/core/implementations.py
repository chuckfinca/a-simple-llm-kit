from typing import Any, List, Union
import dspy
from enum import Enum
from pathlib import Path
import base64
from pydantic import BaseModel, Field, field_validator

from app.core.types import MediaType, PipelineData
from app.core.protocols import ModelBackend
from app.core.model_interfaces import ModelOutput

from typing import Any
from app.core.protocols import ModelBackend
from app.core.model_interfaces import Signature, ModelOutput
import dspy

class DSPyModelBackend:
    """Concrete implementation of ModelBackend protocol"""
    def __init__(self, model_manager, model_id: str, signature_class: type[Signature]):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature_class
    
    async def predict(self, input_data: Any) -> ModelOutput:
        with self.model_manager.get_model(self.model_id) as lm:
            dspy.configure(lm=lm)
            predictor = dspy.Predict(self.signature, lm)
            
            # Let each signature determine its input format
            input_key = "image" if self.signature.__name__ == 'BusinessCardExtractor' else "input"
            raw_result = predictor(**{input_key: input_data})
            
            return self.signature.process_output(raw_result)


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

class ImageType(Enum):
    BASE64 = "base64"
    PNG = "png"
    JPEG = "jpeg"
    NONE = "none"
    
    def to_mime_type(self) -> str:
        """Convert image type to MIME type string"""
        mime_types = {
            ImageType.PNG: "image/png",
            ImageType.JPEG: "image/jpeg",
            ImageType.BASE64: "image/png",  # Default to PNG for base64
            ImageType.NONE: "application/octet-stream"
        }
        return mime_types[self]

class ImageTypeValidator:
    """Validates and detects image types in the pipeline"""
    
    def __init__(self):
        self._accepted_types = [MediaType.IMAGE]
    
    @property
    def accepted_media_types(self) -> List[MediaType]:
        """Implement accepted_media_types as required by Processor protocol"""
        return self._accepted_types
    
    async def process(self, data: PipelineData) -> PipelineData:
        """Implement process method as required by Processor protocol"""
        # Validate the image type
        image_type = self.detect_type(data.content)
        
        # Return data with updated metadata
        return PipelineData(
            media_type=MediaType.IMAGE,
            content=data.content,
            metadata={
                **data.metadata,
                'detected_image_type': image_type.value,
                'validated': True
            }
        )
    
    @staticmethod
    def detect_type(content: Union[str, bytes]) -> ImageType:
        if isinstance(content, bytes):
            return ImageTypeValidator._detect_from_bytes(content)
        return ImageTypeValidator._detect_from_str(content)
            
    @staticmethod
    def _detect_from_bytes(data: bytes) -> ImageType:
        if data.startswith(b'\x89PNG\r\n'):
            return ImageType.PNG
        if data.startswith(b'\xff\xd8\xff'):
            return ImageType.JPEG
        raise ValueError("Invalid image bytes")
    
    @staticmethod
    def _detect_from_str(data: str) -> ImageType:
        path = Path(data)
        if path.exists():
            with open(path, 'rb') as f:
                return ImageTypeValidator._detect_from_bytes(f.read(4))
                
        try:
            decoded = base64.b64decode(data)
            if ImageTypeValidator._detect_from_bytes(decoded):
                return ImageType.BASE64
        except:
            raise ValueError("Must be valid file path or base64 string")


class ImageInput(BaseModel):
    content: Union[str, bytes] 
    type: ImageType = Field(default_factory=lambda: ImageType.NONE)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.type = ImageTypeValidator.detect_type(self.content)
        

class ImageConverterStep:
    """Pipeline step that validates and converts images to the required format"""
    
    @property
    def accepted_media_types(self) -> List[MediaType]:
        return [MediaType.IMAGE]
    
    async def process(self, data: PipelineData) -> PipelineData:
        # Validate and determine image type
        image_input = ImageInput(content=data.content)
        
        # Convert to base64 if not already
        if image_input.type != ImageType.BASE64:
            if isinstance(image_input.content, str):
                # Read file content if path provided
                with open(image_input.content, 'rb') as f:
                    image_bytes = f.read()
            else:
                # Use bytes directly
                image_bytes = image_input.content
                
            # Convert to base64
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
        else:
            base64_data = image_input.content
        
        mime_type = image_input.type.to_mime_type()
        data_uri = f"data:{mime_type};base64,{base64_data}"

        return PipelineData(
            media_type=MediaType.IMAGE,
            content=data_uri,  # Complete data URI
            metadata={
                **data.metadata,
                'original_type': image_input.type.value,
                'mime_type': mime_type,  # New field
                'converted_to': 'data_uri'
            }
        )