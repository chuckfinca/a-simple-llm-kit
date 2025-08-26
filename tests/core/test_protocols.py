from typing import Any
from unittest.mock import MagicMock

import dspy
import pytest

from a_simple_llm_kit.core import ImageProcessor, ModelProcessor, Pipeline, PipelineStep
from a_simple_llm_kit.core.output_processors import DefaultOutputProcessor
from a_simple_llm_kit.core.protocols import ModelBackend
from a_simple_llm_kit.core.types import (
    ImageProcessingMetadata,
    MediaType,
    PipelineData,
    ProgramMetadata,
    Usage,
)


# Test Data
@pytest.fixture
def text_data():
    return PipelineData(media_type=MediaType.TEXT, content="test content", metadata={})


@pytest.fixture
def image_data():
    return PipelineData(
        media_type=MediaType.IMAGE, content=b"fake image bytes", metadata={}
    )


# Mock Implementations
class MockPipelineStep:
    """Simple processor that adds a prefix to text content"""

    def __init__(self, prefix: str = "processed_"):
        self.prefix = prefix
        self._accepted_types = [MediaType.TEXT]

    async def process(self, data: PipelineData) -> PipelineData:
        return PipelineData(
            media_type=MediaType.TEXT,
            content=f"{self.prefix}{data.content}",
            metadata={**data.metadata, "processed": True},
        )

    @property
    def accepted_media_types(self):
        return self._accepted_types


class MockModelBackend:
    """Simple mock model that appends text"""

    program_metadata: ProgramMetadata | None = None
    model_id: str = "mock-model"
    last_prompt_tokens: int | None = 0
    last_completion_tokens: int | None = 0

    async def predict(self, input: str, pipeline_data: PipelineData) -> str:
        return f"{input}_predicted"

    def get_lm_history(self) -> list[Any]:
        return []


# Protocol Conformance Tests
def test_processor_protocol_conformance():
    """Test that our implementations properly satisfy the PipelineStep protocol"""
    processor: PipelineStep = MockPipelineStep()
    assert hasattr(processor, "process")
    assert hasattr(processor, "accepted_media_types")


def test_model_backend_protocol_conformance():
    """Test that our model backend implements the ModelBackend protocol"""
    backend: ModelBackend = MockModelBackend()
    assert hasattr(backend, "predict")


# Pipeline Tests
@pytest.mark.anyio
async def test_pipeline_single_processor(text_data):
    """Test pipeline with a single processor"""
    processor = MockPipelineStep()
    pipeline = Pipeline([processor])
    result = await pipeline.execute(text_data)
    assert result.content == "processed_test content"
    assert result.metadata["processed"] is True


@pytest.mark.anyio
async def test_pipeline_multiple_processors(text_data):
    """Test pipeline with multiple processors in sequence"""
    processors = [MockPipelineStep("first_"), MockPipelineStep("second_")]
    pipeline = Pipeline(processors)
    result = await pipeline.execute(text_data)
    assert result.content == "second_first_test content"


@pytest.mark.anyio
async def test_pipeline_validation():
    """Test that pipeline validates media type compatibility"""
    text_processor = MockPipelineStep()
    image_processor = ImageProcessor()
    with pytest.raises(ValueError):
        Pipeline([text_processor, image_processor])


# Implementation Tests
class TestModelProcessor:
    @pytest.mark.anyio
    async def test_model_processor_basic(self, text_data, monkeypatch):
        """Test basic model processor functionality AND its metadata output."""
        # ARRANGE
        mock_model_manager = MagicMock()
        # Mock the LM history to control the usage data that gets extracted
        mock_lm = MagicMock(spec=dspy.LM)
        mock_lm.history = [{"usage": {"prompt_tokens": 50, "completion_tokens": 100}}]
        mock_model_manager.get_model.return_value = mock_lm

        mock_predictor_instance = MagicMock()
        mock_prediction_result = MagicMock()
        mock_prediction_result.output = "test content_predicted"
        mock_predictor_instance.return_value = mock_prediction_result

        # Create a named mock for the dspy.Predict class
        mock_dspy_predict_class = MagicMock(return_value=mock_predictor_instance)

        # Patch using the named mock
        monkeypatch.setattr("dspy.Predict", mock_dspy_predict_class)
        monkeypatch.setattr("dspy.configure", MagicMock())

        processor = ModelProcessor(
            model_manager=mock_model_manager,
            model_id="gpt-4o-mini",
            signature_class=dspy.Signature,
            input_key="input",
            output_processor=DefaultOutputProcessor(),
            accepted_types=[MediaType.TEXT],
            output_type=MediaType.TEXT,
        )

        # ACT
        result = await processor.process(text_data)

        # ASSERT
        assert result.content == "test content_predicted"
        assert result.metadata["processed"] is True

        assert "usage" in result.metadata
        usage_metadata = result.metadata["usage"]
        assert isinstance(usage_metadata, Usage)
        assert usage_metadata.prompt_tokens == 50
        assert usage_metadata.completion_tokens == 100

    def test_model_processor_media_types(self):
        """Test model processor media type handling"""
        mock_model_manager = MagicMock()
        mock_output_processor = DefaultOutputProcessor()

        processor = ModelProcessor(
            model_manager=mock_model_manager,
            model_id="mock-model-id",
            signature_class=dspy.Signature,
            input_key="input",
            output_processor=mock_output_processor,
            accepted_types=[MediaType.TEXT, MediaType.IMAGE],
            output_type=MediaType.TEXT,
        )

        assert processor.accepted_media_types == [MediaType.TEXT, MediaType.IMAGE]


class TestImageProcessor:
    @pytest.mark.anyio
    async def test_image_processing(
        self,
    ):  # Removed image_data fixture, we create it here
        """Test image processor creates correct, typed metadata."""
        # ARRANGE
        import io

        from PIL import Image

        test_image = Image.new("RGB", (1000, 1200))  # Non-square for better ratio test
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format="PNG")
        data = PipelineData(
            media_type=MediaType.IMAGE, content=img_byte_arr.getvalue(), metadata={}
        )
        processor = ImageProcessor(max_size=(800, 800))

        # ACT
        result = await processor.process(data)

        # ASSERT
        # 1. Check that our specific metadata object exists
        assert "image_processing" in result.metadata
        metadata = result.metadata["image_processing"]
        assert isinstance(metadata, ImageProcessingMetadata)

        # 2. Assert against the typed attributes of the object
        assert metadata.processed is True
        assert metadata.original_size == (1000, 1200)
        # Max size is 800, ratio is 800/1200 = 0.666...
        # New size will be (1000 * 0.666, 1200 * 0.666) = (666, 800)
        assert metadata.processed_size == (666, 800)

    def test_image_processor_media_types(self):
        """Test image processor media type handling"""
        processor = ImageProcessor()
        assert MediaType.IMAGE in processor.accepted_media_types
