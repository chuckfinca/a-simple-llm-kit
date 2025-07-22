from typing import Any, Optional

import pytest

from llm_server.core.implementations import ImageProcessor, ModelProcessor
from llm_server.core.pipeline import Pipeline, PipelineStep
from llm_server.core.protocols import ModelBackend
from llm_server.core.types import MediaType, PipelineData, ProgramMetadata


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

    program_metadata: Optional[ProgramMetadata] = None

    model_id: str = "mock-model"

    last_prompt_tokens: Optional[int] = 0
    last_completion_tokens: Optional[int] = 0

    async def predict(self, input: str) -> str:
        return f"{input}_predicted"

    # ADD a dummy get_lm_history method to conform to the protocol
    def get_lm_history(self) -> list[Any]:
        return []


# Protocol Conformance Tests
def test_processor_protocol_conformance():
    """Test that our implementations properly satisfy the PipelineStep protocol"""
    # This will fail type checking if MockPipelineStep doesn't implement Protocol
    processor: PipelineStep = MockPipelineStep()
    assert hasattr(processor, "process")
    assert hasattr(processor, "accepted_media_types")


def test_model_backend_protocol_conformance():
    """Test that our model backend implements the ModelBackend protocol"""
    backend: ModelBackend = MockModelBackend()
    assert hasattr(backend, "predict")


# Pipeline Tests
@pytest.mark.asyncio
async def test_pipeline_single_processor(text_data):
    """Test pipeline with a single processor"""
    processor = MockPipelineStep()
    pipeline = Pipeline([processor])

    result = await pipeline.execute(text_data)

    assert result.content == "processed_test content"
    assert result.metadata["processed"] is True


@pytest.mark.asyncio
async def test_pipeline_multiple_processors(text_data):
    """Test pipeline with multiple processors in sequence"""
    processors = [MockPipelineStep("first_"), MockPipelineStep("second_")]
    pipeline = Pipeline(processors)

    result = await pipeline.execute(text_data)

    assert result.content == "second_first_test content"


@pytest.mark.asyncio
async def test_pipeline_validation():
    """Test that pipeline validates media type compatibility"""
    text_processor = MockPipelineStep()
    image_processor = ImageProcessor()

    # Should raise ValueError due to incompatible media types
    with pytest.raises(ValueError):
        Pipeline([text_processor, image_processor])


# Implementation Tests
class TestModelProcessor:
    @pytest.mark.asyncio
    async def test_model_processor_basic(self, text_data):
        """Test basic model processor functionality"""
        backend = MockModelBackend()
        processor = ModelProcessor(
            backend=backend, accepted_types=[MediaType.TEXT], output_type=MediaType.TEXT
        )

        result = await processor.process(text_data)

        assert result.content == "test content_predicted"
        assert result.metadata["processed"] is True

    def test_model_processor_media_types(self):
        """Test model processor media type handling"""
        backend = MockModelBackend()
        processor = ModelProcessor(
            backend=backend, accepted_types=[MediaType.TEXT], output_type=MediaType.TEXT
        )

        assert MediaType.TEXT in processor.accepted_media_types


class TestImageProcessor:
    @pytest.mark.asyncio
    async def test_image_processing(
        self, image_data
    ):  # The parameter name is fine, but the fixture data is unused
        """Test image processor functionality"""
        import io

        from PIL import Image

        # Create a real test image
        test_image = Image.new("RGB", (1000, 1000))
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format="PNG")

        data = PipelineData(
            media_type=MediaType.IMAGE, content=img_byte_arr.getvalue(), metadata={}
        )

        processor = ImageProcessor(max_size=(800, 800))
        result = await processor.process(data)

        # Verify the processed image size from metadata
        assert "processed_size" in result.metadata
        assert result.metadata["processed_size"] == (800, 800)
        assert result.metadata["processed"] is True

    def test_image_processor_media_types(self):
        """Test image processor media type handling"""
        processor = ImageProcessor()
        assert MediaType.IMAGE in processor.accepted_media_types
