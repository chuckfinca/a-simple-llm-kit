import pytest
import os
from unittest.mock import MagicMock, AsyncMock
from app.core.pipeline import Pipeline
from app.core.types import MediaType, PipelineData
from app.services.pipelines.predictor import PredictorStep
from app.models.predictor import Predictor

@pytest.fixture
def mock_model_manager():
    manager = MagicMock()
    # Mock the context manager
    manager.get_model.return_value.__enter__.return_value = "mock_lm"
    manager.get_model.return_value.__exit__.return_value = None
    return manager

@pytest.mark.asyncio
async def test_predictor_pipeline_mock(mock_model_manager):
    # Create pipeline with predictor step
    pipeline = Pipeline([PredictorStep(mock_model_manager, "gpt-4")])

    # Create initial data
    initial_data = PipelineData(
        media_type=MediaType.TEXT,
        content="Complete this: The quick brown fox...",
        metadata={"test": True}
    )

    # Mock the predictor response
    mock_response = MagicMock()
    mock_response.output = "jumps over the lazy dog"

    # Configure the mock predictor - no need for AsyncMock since it's synchronous
    predictor_mock = MagicMock()
    predictor_mock.return_value = mock_response

    # Create a mock Predict class that returns our mock predictor
    predict_class_mock = MagicMock(return_value=predictor_mock)

    with pytest.MonkeyPatch.context() as mp:
        # Mock both dspy.configure and dspy.Predict
        mp.setattr("dspy.configure", MagicMock())
        mp.setattr("dspy.Predict", predict_class_mock)

        # Execute pipeline
        result = await pipeline.execute(initial_data)

    # Assertions
    assert result.media_type == MediaType.TEXT
    assert result.content == "jumps over the lazy dog"
    assert result.metadata["model_used"] == "gpt-4"
    assert result.metadata["test"] is True

    # Verify the predictor was called correctly
    predict_class_mock.assert_called_once_with(Predictor, "mock_lm")

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
@pytest.mark.asyncio
async def test_predictor_pipeline_integration():
    """Integration test using real OpenAI API"""
    from app.models.manager import ModelManager
    from app.core.config import Settings
    
    # Initialize real model manager
    settings = Settings()
    model_manager = ModelManager(settings.config_path)
    
    # Create pipeline with predictor step
    pipeline = Pipeline([PredictorStep(model_manager, "gpt-4")])
    
    # Create test data
    initial_data = PipelineData(
        media_type=MediaType.TEXT,
        content="Complete this: The quick brown fox...",
        metadata={"test": True}
    )
    
    try:
        # Execute pipeline
        result = await pipeline.execute(initial_data)
        
        # Basic assertions
        assert result.media_type == MediaType.TEXT
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert "dog" in result.content.lower()  # Common completion
        assert result.metadata["model_used"] == "gpt-4"
        assert result.metadata["test"] is True
    except Exception as e:
        pytest.fail(f"Integration test failed with error: {str(e)}")