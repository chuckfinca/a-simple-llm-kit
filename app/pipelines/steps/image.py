from PIL import Image
import io
from typing import List, Any

from app.core.types import MediaType, PipelineData

class ImagePreprocessor(PipelineStep):
    """Handles image preprocessing like resizing, normalization."""
    @property
    def accepted_media_types(self) -> List[MediaType]:
        return [MediaType.IMAGE]

    async def process(self, data: PipelineData) -> PipelineData:
        image_bytes = data.content
        image = Image.open(io.BytesIO(image_bytes))
        
        # Add preprocessing logic here
        # Example: Resize image
        processed_image = image.resize((800, 800))
        
        # Convert back to bytes
        buffer = io.BytesIO()
        processed_image.save(buffer, format="PNG")
        
        return PipelineData(
            media_type=MediaType.IMAGE,
            content=buffer.getvalue(),
            metadata={**data.metadata, "preprocessed": True}
        )

class OCRProcessor(PipelineStep):
    """Extracts text from images."""
    @property
    def accepted_media_types(self) -> List[MediaType]:
        return [MediaType.IMAGE]

    async def process(self, data: PipelineData) -> PipelineData:
        # Add OCR logic here
        # This would convert image to text
        extracted_text = "Sample extracted text"
        
        return PipelineData(
            media_type=MediaType.TEXT,
            content=extracted_text,
            metadata={**data.metadata, "ocr_applied": True}
        )
