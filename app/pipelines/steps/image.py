from PIL import Image
import io
from typing import Tuple
from app.pipelines.steps.base import BaseImageStep
from app.core.types import MediaType, PipelineData

class ImagePreprocessor(BaseImageStep):
    """Handles basic image preprocessing operations."""
    def __init__(self, target_size: Tuple[int, int] = (800, 800)):
        self.target_size = target_size

    async def process(self, data: PipelineData) -> PipelineData:
        image_bytes = data.content
        image = Image.open(io.BytesIO(image_bytes))
        
        processed_image = image.resize(self.target_size)
        
        buffer = io.BytesIO()
        processed_image.save(buffer, format="PNG")
        
        return PipelineData(
            media_type=MediaType.IMAGE,
            content=buffer.getvalue(),
            metadata={**data.metadata, "preprocessed": True}
        )