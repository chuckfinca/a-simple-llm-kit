import pytest

from llm_server.core.factories import create_text_processor
from llm_server.core.protocols import PipelineStep
from llm_server.core.types import MediaType


@pytest.fixture
def mock_model_manager(monkeypatch):
    """Create a mock model manager for testing"""

    class MockLM:
        """A minimal mock for the language model."""
        pass

    class MockModelManager:
        """A mock that correctly has a .models attribute."""
        def __init__(self):
            self.models = {"test-model": MockLM()}

    # --- THIS IS THE FIX ---
    # We create a class to mock dspy.Predict
    class MockPredict:
        def __init__(self, signature):
            """Mocks the instantiation of dspy.Predict(signature)"""
            # We don't need to use the signature in this test, but we
            # accept it to match the real class.
            pass

        def __call__(self, **kwargs):
            """Mocks the call to the predictor instance, e.g., predictor(input=...)"""
            input_value = kwargs.get("input", "")
            # Return a simple object with an 'output' attribute.
            return type("obj", (), {"output": f"{input_value}_processed"})()

    # Replace the actual dspy.Predict class with our mock class
    monkeypatch.setattr("dspy.Predict", MockPredict)

    # We still need to mock dspy.configure as it's called in the code
    monkeypatch.setattr("dspy.configure", lambda lm: None)

    return MockModelManager()



def test_create_text_processor(mock_model_manager):
    """Test creation of text processor"""
    processor = create_text_processor(mock_model_manager, "test-model")

    # Verify it implements the PipelineStep protocol
    assert isinstance(processor, PipelineStep)
    assert MediaType.TEXT in processor.accepted_media_types


@pytest.mark.asyncio
async def test_text_processor_functionality(mock_model_manager):
    """Test the created text processor actually works"""
    from llm_server.core.types import PipelineData

    processor = create_text_processor(mock_model_manager, "test-model")

    result = await processor.process(
        PipelineData(media_type=MediaType.TEXT, content="test input", metadata={})
    )

    assert result.content == "test input_processed"
    assert result.metadata.get("processed") is True
