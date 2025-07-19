# DSPy Program Tracking System

This document provides an overview of the DSPy program tracking system integrated into the LLM Server. This system enables tracking, versioning, and evaluation of DSPy programs without requiring separate model version tracking.

## Core Concepts

### Programs and Versions

- **Program**: A DSPy Signature class that defines a specific task (e.g., text completion, contact extraction)
- **Program Version**: A specific iteration of a program, using semantic versioning (e.g., 1.0.0, 1.1.0)
- **Program Tree**: Hierarchical structure showing the evolution of programs and their optimized variants

### Execution Tracking

- **Execution**: A single run of a program with a specific model, tracked with metadata
- **Evaluation**: Performance metrics for a specific program-model pair

## Getting Started

### 1. Register a DSPy Program

```python
from llm_server.models.predictor import Predictor

# Get program manager from app state
program_manager = request.app.state.program_manager

# Register a program
metadata = program_manager.register_program(
    program_class=Predictor,
    name="Text Completion",
    description="Basic text completion signature",
    tags=["text", "completion"]
)

print(f"Program ID: {metadata.id}")
```

### 2. Execute a Program with Tracking

```python
# Execute and track
result, execution_info = await program_manager.execute_program(
    program_id=metadata.id,
    model_id="gpt-4o-mini",
    input_data={"input": "Explain quantum computing in simple terms"}
)

print(f"Execution ID: {execution_info.execution_id}")
print(f"Model used: {execution_info.model_id}")
print(f"Result: {result.output}")
```

### 3. Register an Optimized Program

```python
# After optimizing a program with a DSPy optimizer
optimized_metadata = program_manager.register_optimized_program(
    program_class=OptimizedPredictor,
    parent_id=metadata.id,
    optimizer_name="dspy.teleprompt.BootstrapFewShot",
    description="Optimized with bootstrapped few-shot examples"
)
```

### 4. Save Evaluation Results

```python
# Save evaluation results
program_manager.save_evaluation_result(
    program_id=metadata.id,
    model_id="gpt-4o-mini",
    results={
        "accuracy": 0.95,
        "latency_ms": 250,
        "test_cases": 100
    }
)
```

## Factory Methods for Pipeline Creation

The system provides enhanced factory methods that automatically track program usage:

```python
from llm_server.core.factories import create_text_processor

# Create a processor with tracking
processor = create_text_processor(
    model_manager=app.state.model_manager,
    model_id="gpt-4o-mini",
    program_manager=app.state.program_manager,
    metadata={"purpose": "text-summarization"}
)

# Add to a pipeline
pipeline = Pipeline([processor])
```

## REST API Endpoints

The system provides REST API endpoints for accessing program information:

### Program Management

- `GET /v1/programs` - List all registered programs
- `GET /v1/programs/{program_id}` - Get details for a specific program
- `GET /v1/programs/{program_id}/versions` - List all versions of a program
- `GET /v1/programs/{program_id}/evaluations` - Get evaluation results for a program

### Execution Tracking

- `GET /v1/programs/executions` - Get execution history, optionally filtered

### Model Information

- `GET /v1/programs/models` - List all available models from config

## Directory Structure

The system stores program data in the following directory structure:

```
dspy_programs/
├── text_completion/             # Program ID
│   ├── 1.0.0.json               # Version metadata
│   ├── 1.1.0.json               # Updated version
│   └── evaluations/             
│       ├── eval1_gpt-4o-mini.json  # Evaluation results
│       └── ...
├── contact_extractor/
│   └── ...
└── ...
```

## Creating Custom DSPy Programs

When creating custom DSPy programs for tracking:

1. Define your program as a DSPy Signature class:

```python
import dspy

class CustomExtractor(dspy.Signature):
    """Extract specific information from text."""
    input: str = dspy.InputField()
    entity: str = dspy.OutputField()
    attributes: list = dspy.OutputField()
    
    @classmethod
    def process_output(cls, result):
        # Process and validate the output
        return {
            "entity": result.entity,
            "attributes": result.attributes
        }
```

2. Register and use the program:

```python
# Register the program
metadata = program_manager.register_program(
    program_class=CustomExtractor,
    name="Entity Extractor",
    description="Extracts entities and their attributes from text"
)

# Execute with tracking
result, execution_info = await program_manager.execute_program(
    program_id=metadata.id,
    model_id="gpt-4o-mini",
    input_data={"input": "Sample text to process"}
)
```

## Implementation Details

The system consists of several key components:

1. **ProgramRegistry**: Manages DSPy program signatures, their versions, and metadata
2. **ProgramManager**: High-level interface for working with programs and tracking executions
3. **Enhanced ModelProcessor and DSPyModelBackend**: Automatically tracks program executions

All tracked data is persisted to disk in JSON format, allowing it to be backed up, committed to version control, or transferred between environments.

## Storage and Persistence

For persistent storage between restarts:

- The program data is stored in the `dspy_programs` directory within your project
- For Docker deployments, mount this directory as a volume:
  ```yaml
  volumes:
    - ./dspy_programs:/app/dspy_programs
  ```

- For Modal deployments, use a Modal volume:
  ```python
  # In deploy_modal_app.py
  VOLUME_NAME = f"llm-server-{ENV_NAME}-programs"
  volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
  
  # Then mount in your function
  @app.function(
      volumes={"/app/dspy_programs": volume}
  )
  ```

## Best Practices

- **Use Meaningful Names**: Give programs descriptive names that reflect their purpose
- **Add Tags**: Use tags to categorize programs for easier filtering
- **Version Carefully**: Follow semantic versioning when creating new program versions
- **Document Programs**: Add detailed descriptions to programs to explain their purpose and behavior
- **Track Optimizations**: Always register optimized versions with references to their parent programs
- **Store Evaluations**: Save evaluation results for different program-model combinations to track performance