import pytest
from app.core.pipeline import Pipeline, PipelineStep
from app.core.types import MediaType, PipelineData

class SimpleTestStep(PipelineStep):
    """Test step that adds a prefix to text."""
    def __init__(self, prefix: str = "test_"):
        self.prefix = prefix

    @property
    def accepted_media_types(self):
        return [MediaType.TEXT]

    async def process(self, data: PipelineData) -> PipelineData:
        return PipelineData(
            media_type=MediaType.TEXT,
            content=f"{self.prefix}{data.content}",
            metadata={**data.metadata, "processed": True}
        )

@pytest.fixture
def simple_pipeline():
    return Pipeline([SimpleTestStep()])

@pytest.mark.asyncio
async def test_pipeline_execution(simple_pipeline):
    # Given
    initial_data = PipelineData(
        media_type=MediaType.TEXT,
        content="hello",
        metadata={}
    )
    
    # When
    result = await simple_pipeline.execute(initial_data)
    
    # Then
    assert result.content == "test_hello"
    assert result.metadata["processed"] is True

@pytest.mark.asyncio
async def test_pipeline_validation():
    # Given
    class ImageStep(PipelineStep):
        @property
        def accepted_media_types(self):
            return [MediaType.IMAGE]
        
        async def process(self, data):
            return data
    
    # When/Then
    with pytest.raises(ValueError):
        Pipeline([SimpleTestStep(), ImageStep()])  # Should fail validation