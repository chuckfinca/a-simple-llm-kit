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
    def __init__(self, model_manager, model_id: str, signature):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature
    
    async def predict(self, input: Any) -> Any:
        with self.model_manager.get_model(self.model_id) as lm:
            dspy.configure(lm=lm)
            predictor = dspy.Predict(self.signature, lm)
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

class ImageInput(BaseModel):
    content: Union[str, bytes]
    type: ImageType = Field(default=ImageType.NONE)
    
    @field_validator("content")
    @classmethod
    def validate_image(cls, v: Union[str, bytes], info) -> Union[str, bytes]:
        image_type = None
        
        if isinstance(v, bytes):
            if v.startswith(b'\x89PNG\r\n'):
                image_type = ImageType.PNG
            elif v.startswith(b'\xff\xd8\xff'):
                image_type = ImageType.JPEG
            else:
                raise ValueError("Invalid image bytes")
                
        elif isinstance(v, str):
            path = Path(v)
            if path.exists():
                with open(path, 'rb') as f:
                    header = f.read(4)
                    if header.startswith(b'\x89PNG'):
                        image_type = ImageType.PNG
                    elif header.startswith(b'\xff\xd8\xff'):
                        image_type = ImageType.JPEG
                    else:
                        raise ValueError("Not a PNG/JPEG file")
            else:
                try:
                    decoded = base64.b64decode(v)
                    if decoded.startswith((b'\x89PNG', b'\xff\xd8\xff')):
                        image_type = ImageType.BASE64
                except:
                    raise ValueError("Must be valid file path or base64 string")
                    
        if not image_type:
            raise ValueError("Input must be string or bytes")
            
        info.context["type"] = image_type
        return v
    
    @field_validator("type", mode="before")
    @classmethod
    def set_type(cls, v, info):
        return info.context.get("type", ImageType.NONE)
    

class ImageValidator:
    """Validates image input format"""
    @property
    def accepted_media_types(self) -> list[MediaType]:
        return [MediaType.IMAGE]
        
    async def process(self, data: PipelineData) -> PipelineData:
        validated = ImageInput(content=data.content)
        return PipelineData(
            media_type=MediaType.IMAGE,
            content=data.content,
            metadata={**data.metadata, "image_type": validated.type.value}
        )
        

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