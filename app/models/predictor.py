import dspy

class Predictor(dspy.Signature):
    """Simple prediction signature for basic completion."""
    input = dspy.InputField()
    output = dspy.OutputField()