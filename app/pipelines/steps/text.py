from typing import List
from app.pipelines.steps.base import BaseDSPyStep
from app.core.types import MediaType
import dspy

class TextCompletionStep(BaseDSPyStep):
    """Basic text completion using DSPy Predictor."""
    @property
    def accepted_media_types(self) -> List[MediaType]:
        return [MediaType.TEXT]
        
    @property
    def dspy_module(self):
        from app.models.predictor import Predictor
        return Predictor

class DSPyOCRStep(BaseDSPyStep):
    """OCR using DSPy and vision-language models."""
    @property
    def accepted_media_types(self) -> List[MediaType]:
        return [MediaType.IMAGE]
        
    @property
    def dspy_module(self):
        class OCRModule(dspy.Module):
            def forward(self, input: bytes) -> str:
                return self.generate(
                    instruction="Extract all text visible in this image.",
                    input={"image": input}
                )
        return OCRModule
