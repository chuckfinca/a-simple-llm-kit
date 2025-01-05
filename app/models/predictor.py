import dspy
from typing import Any
from app.core.model_interfaces import TextOutput

class Predictor(dspy.Signature):
    """Basic text prediction"""
    input: str = dspy.InputField()
    output: str = dspy.OutputField()
    
    @classmethod
    def process_output(cls, result: Any) -> TextOutput:
        return TextOutput(text=result.output)

