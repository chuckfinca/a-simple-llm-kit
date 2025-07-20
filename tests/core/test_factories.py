import pytest

from llm_server.core.factories import create_text_processor
from llm_server.core.protocols import PipelineStep
from llm_server.core.types import MediaType


@pytest.fixture
def mock_model_manager(monkeypatch):
    """Create a mock model manager for testing"""

    class MockLM:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockModelManager:
        def get_model(self, model_id):
            return MockLM()

    # Mock dspy configuration
    monkeypatch.setattr("dspy.configure", lambda lm: None)
    monkeypatch.setattr(
        "dspy.Predict",
        lambda module, lm: lambda input: type(
            "obj", (), {"output": f"{input}_processed"}
        )(),
    )

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
