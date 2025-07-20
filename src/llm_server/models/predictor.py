import dspy
from typing import Any
from llm_server.core.model_interfaces import SimpleOutput


class Predictor(dspy.Signature):
    """Basic text prediction"""

    input: str = dspy.InputField()
    output: str = dspy.OutputField()

    @classmethod
    def process_output(cls, result: Any) -> SimpleOutput:
        return SimpleOutput(result.output)
