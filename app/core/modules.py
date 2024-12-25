import dspy

class OCRModule(dspy.Module):
    """DSPy module for OCR tasks"""
    def forward(self, input: bytes) -> str:
        return self.generate(
            instruction="Extract all text visible in this image.",
            input={"image": input}
        )
