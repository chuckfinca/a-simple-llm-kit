from typing import Any

import dspy

from a_simple_llm_kit.core.model_interfaces import SimpleOutput


class Predictor(dspy.Signature):
    """Basic text prediction"""

    input: str = dspy.InputField()
    output: str = dspy.OutputField()

    @classmethod
    def process_output(cls, result: Any) -> SimpleOutput:
        return SimpleOutput(result.output)
