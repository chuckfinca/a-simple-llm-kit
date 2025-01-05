from typing import Any, List, Union
import dspy
from enum import Enum
from pathlib import Path
import base64
from pydantic import BaseModel, Field, field_validator

from app.core.types import MediaType, PipelineData
from app.core.protocols import ModelBackend

class DSPyBackend:
    """DSPy-based model implementation"""
    def __init__(self, model_manager, model_id: str, signature_class):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature_class = signature_class
    
    async def predict(self, input: Any) -> Any:
        with self.model_manager.get_model(self.model_id) as lm:
            dspy.configure(lm=lm)
            predictor = dspy.Predict(self.signature_class, lm)
            print(input)
            if self.signature_class.__name__ == 'BusinessCardExtractor':
                result = predictor(image=input)
            else:
                result = predictor(input=input)
            return result.output

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
            base64_content = base64.b64encode(image_bytes).decode('utf-8')
        else:
            base64_content = image_input.content
            
        return PipelineData(
            media_type=MediaType.IMAGE,
            content=base64_content,
            metadata={
                **data.metadata,
                'original_type': image_input.type.value,
                'converted_to': 'base64'
            }
        )