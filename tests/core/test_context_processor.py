from unittest.mock import MagicMock

import dspy
import pytest

from a_simple_llm_kit.core import MediaType, PipelineData, TaskContext
from a_simple_llm_kit.core.implementations import ContextModelProcessor
from a_simple_llm_kit.core.output_processors import DefaultOutputProcessor
from a_simple_llm_kit.models.signatures import ContextSignature

# --- Fixtures ---


@pytest.fixture
def task_context():
    return TaskContext(context_data="test document data", chat_history="User: Hi")


@pytest.fixture
def context_pipeline_data(task_context):
    return PipelineData(
        media_type=MediaType.TEXT,
        content=task_context,
        metadata={"input_args": {"topic": "Finance"}},
    )


class MockContextSignature(ContextSignature):
    """A mock signature for testing the processor"""

    topic: str = dspy.InputField()
    output: str = dspy.OutputField()


# --- Tests ---


def test_task_context_structure():
    """Unit test for the TaskContext data container"""
    ctx = TaskContext(context_data="test context", chat_history="Chat")

    assert ctx.context_data == "test context"
    assert ctx.chat_history == "Chat"


@pytest.mark.anyio
async def test_context_processor_success(context_pipeline_data, monkeypatch):
    """
    Integration test validating:
    1. Context injection works
    2. Specific kwargs (topic) are passed
    3. Threading wrapper works
    4. Usage is extracted
    """
    # 1. Mock the Model Manager & LM
    mock_model_manager = MagicMock()
    mock_lm = MagicMock(spec=dspy.LM)
    # Simulate history for usage extraction
    mock_lm.history = [{"usage": {"input_tokens": 10, "output_tokens": 20}}]
    mock_model_manager.get_model.return_value = mock_lm

    # 2. Mock DSPy ChainOfThought & Predictor
    mock_predictor_instance = MagicMock()
    # The result object returned by the predictor
    mock_prediction_result = MagicMock()
    mock_prediction_result.output = "Generated Report"
    mock_predictor_instance.return_value = mock_prediction_result

    # Mock the class instantiation
    mock_cot_class = MagicMock(return_value=mock_predictor_instance)

    # 3. Apply Monkeypatches to intercept DSPy calls
    monkeypatch.setattr("dspy.ChainOfThought", mock_cot_class)
    monkeypatch.setattr("dspy.context", MagicMock())

    # 4. Initialize Processor
    processor = ContextModelProcessor(
        model_manager=mock_model_manager,
        model_id="claude-3-5-sonnet",  # Triggers Anthropic usage logic
        signature_class=MockContextSignature,
        output_processor=DefaultOutputProcessor(),
    )

    # 5. Execute
    result = await processor.process(context_pipeline_data)

    # --- ASSERTIONS ---

    # A. Verify Output
    assert result.content == "Generated Report"
    assert result.metadata["processed"] is True

    # B. Verify Usage Extraction (Critical for billing)
    assert "usage" in result.metadata
    usage = result.metadata["usage"]
    # Usage is now serialized as dict for JSON compatibility
    assert isinstance(usage, dict)
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20

    # C. Verify Context Injection
    # Ensure the predictor was called with BOTH the context AND the specific args
    call_kwargs = mock_predictor_instance.call_args[1]

    assert call_kwargs["context_documents"] == "test document data"
    assert call_kwargs["chat_history"] == "User: Hi"
    assert call_kwargs["topic"] == "Finance"  # Came from metadata['input_args']


def test_context_processor_validation_error():
    """Ensure it fails fast if we pass wrong data type"""
    processor = ContextModelProcessor(
        model_manager=MagicMock(),
        model_id="test",
        signature_class=MockContextSignature,
        output_processor=DefaultOutputProcessor(),
    )

    bad_data = PipelineData(
        media_type=MediaType.TEXT,
        content="Just a string, not a TaskContext",
        metadata={},
    )

    # Should run immediately (no async needed for validation check usually,
    # but process is async, so we wrap it)
    import anyio

    async def run_check():
        with pytest.raises(ValueError) as exc:
            await processor.process(bad_data)
        assert "requires TaskContext" in str(exc.value)

    anyio.run(run_check)
