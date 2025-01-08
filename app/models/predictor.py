import dspy
from typing import Any

class Predictor(dspy.Signature):
    """Basic text prediction"""
    input: str = dspy.InputField()
    output: str = dspy.OutputField()

