# LLM Server Framework

A production-ready Python library for building LLM-powered applications with advanced pipeline processing, multi-modal capabilities, and enterprise-grade reliability features. Built with FastAPI, DSPy, and a composable architecture.

**This is a framework/library, not a standalone server.** Use it as a dependency in your own FastAPI applications.

## 🚀 Key Features

### Core Architecture
- **Pipeline-First Design**: Composable, type-safe pipeline steps for complex processing workflows
- **Protocol-Based Framework**: Clean interfaces enabling easy extension and testing
- **Multi-Modal Processing**: Unified handling of text, images, and structured data
- **Performance Tracking**: Comprehensive, per-request metrics collection with step-by-step timing

### Reliability & Observability
- **Circuit Breaker Pattern**: Built-in failure protection with automatic recovery
- **OpenTelemetry Integration**: Vendor-neutral metrics and tracing for any backend (Prometheus, Datadog, etc.)
- **Semantic Conventions**: Adheres to `llm.*` OTel conventions for out-of-the-box compatibility with observability tools
- **Structured Logging**: JSON-formatted logs with context preservation

### Model & Provider Support
- **Multi-Provider**: OpenAI, Anthropic, Google Gemini, and Hugging Face
- **Flexible Configuration**: YAML-based model configuration with parameter overrides
- **Token Management**: Robust and accurate token counting with cost estimation
- **Program Versioning**: DSPy program management with optimization tracking

### Specialized Capabilities
- **Image Processing**: Intelligent resizing, format conversion, and optimization
- **Type Safety**: Full Pydantic integration with runtime protocol checking
- **Custom Extensions**: Easy-to-implement custom pipeline steps and model backends

## 🏗️ Architecture Overview

The framework is built around a composable pipeline architecture where each step implements the `PipelineStep` protocol:

```python
# Core pipeline concept
Pipeline([
    ImageProcessor(max_size=(800, 800)),           # Resize and optimize images
    ModelProcessor(backend, [MediaType.IMAGE]),    # Send to vision model
    OutputProcessor()                              # Format response
])
```

### Key Components

- **Core Framework** (`src/llm_server/core/`): Protocols, types, and base implementations
- **Model Management** (`src/llm_server/models/`): Provider abstraction and program management
- **Pipeline System**: Composable processing steps with automatic validation
- **Metrics & Monitoring**: Performance tracking, circuit breakers, and observability

## 💻 Installation & Basic Usage

### Installation

```bash
# Install from PyPI (when published)
pip install llm-server-framework

# Or install from source for development
git clone https://github.com/chuckfinca/llm-server
cd llm-server
pip install -e ".[dev]"
```

### Basic Usage Example

Here's how to build a simple FastAPI application that uses the framework to provide a text completion endpoint with full performance tracking.

**Your Application's `main.py`:**

```python
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import dspy

# 1. Import the framework's core components
from llm_server.core.config import FrameworkSettings
from llm_server.core.metrics_wrappers import PerformanceMetrics, ModelBackendTracker
from llm_server.defaults import FileSystemStorageAdapter, YamlConfigProvider
from llm_server.models.manager import ModelManager
from llm_server.models.predictor import Predictor # A basic dspy.Signature
from llm_server.models.program_manager import ProgramManager

class PredictionRequest(BaseModel):
    prompt: str
    model_id: str = "gpt-4o-mini"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 2. Configure and instantiate the framework managers
    settings = FrameworkSettings()
    config_provider = YamlConfigProvider("config/model_config.yml")
    storage_adapter = FileSystemStorageAdapter(base_dir="dspy_programs")

    model_manager = ModelManager(config_provider=config_provider, settings=settings)
    program_manager = ProgramManager(model_manager=model_manager, storage_adapter=storage_adapter)

    # 3. Register your application's DSPy programs
    program_manager.register_program(program_class=Predictor, name="Text Completion")

    # 4. Make managers available to your API routes
    app.state.program_manager = program_manager
    yield
    app.state.program_manager = None

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict_text(request: PredictionRequest):
    """A modern endpoint using the pipeline and automatic metadata."""
    try:
        program_manager: ProgramManager = app.state.program_manager
        metrics = PerformanceMetrics() # 1. Start performance tracking

        # 2. Define the processing pipeline
        pipeline = Pipeline([
            ModelProcessor(
                model_manager=program_manager.model_manager,
                model_id=request.model_id,
                signature_class=Predictor,
                input_key="input",
                output_processor=DefaultOutputProcessor(),
                accepted_types=[MediaType.TEXT],
                output_type=MediaType.TEXT,
            )
        ])

        # 3. Execute the pipeline
        result_data = await pipeline.execute(
            PipelineData(media_type=MediaType.TEXT, content=request.prompt)
        )
        metrics.mark_checkpoint("pipeline_complete")

        # 4. Automatically collect all consistent metadata
        final_metadata = MetadataCollector.collect_response_metadata(
            model_id=request.model_id,
            program_metadata=program_manager.registry.get_program_metadata("predictor"),
            performance_metrics=metrics.get_summary(),
            model_info=program_manager.model_info.get(request.model_id, {}).model_dump(by_alias=True)
        )

        return {
            "success": True,
            "data": {"response": result_data.content},
            "metadata": final_metadata,
        }
    except Exception as e:
        # Proper error handling
        raise HTTPException(status_code=500, detail=str(e))
```

### Required Configuration

**`config/model_config.yml`:**

```yaml
models:
  gpt-4o-mini:
    model_name: "openai/gpt-4o-mini"
    max_tokens: 3000
  claude-3.5-sonnet:
    model_name: "anthropic/claude-3-5-sonnet-20241022"
    max_tokens: 4000
```

**Environment Variables (`.env`):**

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### Running Your Application

```bash
# Run your FastAPI application
uvicorn main:app --reload

# Test your custom endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "model_id": "gpt-4o-mini"}'
```

## 🔭 Observability with OpenTelemetry

The framework is deeply instrumented with OpenTelemetry to provide vendor-neutral metrics and traces, giving you immediate insight into your application's performance.

### Enabling Observability

**1. Install the optional dependencies:**

```bash
pip install "llm-server-framework[opentelemetry]"
```

**2. Enable via Environment Variables:**

Create a `.env` file in your application's root directory:

```env
# --- Enable OTel ---
OTEL_ENABLED=true
OTEL_SERVICE_NAME="MyLLMApp"
OTEL_SERVICE_VERSION="1.0.0"

# --- Your API Keys ---
OPENAI_API_KEY=your_openai_key_here
# ...
```

**3. Configure the OTel SDK in Your Application:**

The library emits signals, but your application is responsible for configuring an "exporter" to send them to a backend. Here is an example of setting up a Prometheus exporter in your `main.py`:

```python
# In your main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager

# --- OTel Imports ---
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import make_asgi_app
# --- End OTel Imports ---

# Import your settings to get service name
from llm_server.core.config import FrameworkSettings

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = FrameworkSettings()
    
    # --- OTel SDK Setup ---
    if settings.otel_enabled:
        resource = Resource.create({"service.name": settings.otel_service_name, "service.version": settings.otel_service_version})
        reader = PrometheusMetricReader()
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
    # --- End OTel SDK Setup ---
    
    # ... your existing lifespan logic for managers ...
    yield

# Create the main FastAPI app
app = FastAPI(lifespan=lifespan)

# Create and mount the Prometheus metrics endpoint
settings = FrameworkSettings()
if settings.otel_enabled:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# ... your API routes (@app.post("/predict"), etc.) ...
```

### Available Instrumentation

- **Automatic Tracing**: Key methods like `ModelBackend.predict` and each step in a Pipeline are automatically wrapped in trace spans with rich, LLM-specific attributes
- **Automatic Metrics**: The framework emits metrics for model calls, circuit breaker failures and state changes, and overall request latency
- **Rich Per-Request Data**: The `PerformanceMetrics` object, accessible in your API response, provides a detailed breakdown of timing and token usage for debugging individual requests

## 📊 Response Format

All API responses follow a consistent envelope with comprehensive metadata:

```json
{
  "success": true,
  "data": {
    "response": "Model response content"
  },
  "metadata": {
    "program": {
      "id": "predictor",
      "version": "1.0.0",
      "name": "Predictor"
    },
    "model": {
      "id": "gpt-4o-mini",
      "provider": "openai",
      "base_model": "gpt-4o-mini"
    },
    "performance": {
      "timing": {
        "total_ms": 750.25,
        "model_complete_ms": 738.91
      },
      "tokens": {
        "input": 50,
        "output": 150,
        "total": 200,
        "cost_usd": 0.0001,
        "method": "dspy_history_exact"
      },
      "trace_id": "unique-trace-identifier"
    },
    "execution_id": "unique-execution-id",
    "timestamp": "2025-08-01T10:30:00.123456Z"
  }
}
```

## 🔧 Building Custom Pipelines

The framework's strength lies in its composable pipeline architecture. Create custom processing steps by implementing the `PipelineStep` protocol:

### Custom Pipeline Step

```python
from llm_server.core.protocols import PipelineStep
from llm_server.core.types import MediaType, PipelineData

class TextSummarizerStep(PipelineStep):
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        
    @property
    def accepted_media_types(self) -> list[MediaType]:
        return [MediaType.TEXT]
        
    async def process(self, data: PipelineData) -> PipelineData:
        # Custom processing logic
        text = data.content
        if len(text) > self.max_length:
            summary = text[:self.max_length] + "..."
        else:
            summary = text
            
        return PipelineData(
            media_type=MediaType.TEXT,
            content=summary,
            metadata={
                **data.metadata,
                "original_length": len(text),
                "summarized": True
            }
        )
```

### Custom Model Backend

```python
from llm_server.core.protocols import ModelBackend
from llm_server.core.model_interfaces import ModelOutput

class CustomModelBackend(ModelBackend):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.program_metadata = None
        self.last_prompt_tokens = None
        self.last_completion_tokens = None
    
    async def predict(self, input: Any) -> ModelOutput:
        # Your custom model logic here
        result = await your_model_call(input)
        return result
    
    def get_lm_history(self) -> list[Any]:
        return []  # Return model interaction history
```

### Combining Custom Components

```python
from llm_server.core.pipeline import Pipeline
from llm_server.core.implementations import ModelProcessor

# Create a custom pipeline
custom_pipeline = Pipeline([
    TextSummarizerStep(max_length=200),
    ModelProcessor(
        backend=CustomModelBackend("my-model"), 
        accepted_types=[MediaType.TEXT],
        output_type=MediaType.TEXT
    )
])

# Execute the pipeline
result = await custom_pipeline.execute(initial_data)
```

## ⚙️ Configuration

### Model Configuration (`config/model_config.yml`)

```yaml
models:
  gpt-4o-mini:
    model_name: "openai/gpt-4o-mini"
    max_tokens: 3000
    additional_params:
      temperature: 0.7
      top_p: 1.0
      
  claude-3.5-sonnet:
    model_name: "anthropic/claude-3-5-sonnet-20241022"
    max_tokens: 4000
    additional_params:
      temperature: 0.8
      
  Meta-Llama-3.1-8B-Instruct:
    model_name: "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_tokens: 3000
    additional_params: {}
    
  gemini-2.0-flash:
    model_name: "gemini/gemini-2.0-flash"
    max_tokens: 2048
    additional_params:
      temperature: 0.9
```

### Environment Variables

```env
# Required API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_hf_key
GEMINI_API_KEY=your_gemini_key

# Optional: Custom config path
LLM_CONFIG_PATH=config/model_config.yml

# OpenTelemetry Configuration
OTEL_ENABLED=true
OTEL_SERVICE_NAME="MyLLMApp"
OTEL_SERVICE_VERSION="1.0.0"
```

## 🔧 Framework Integration Patterns

### Pipeline-Based Processing

Build complex processing workflows using the pipeline architecture:

```python
from llm_server.core.pipeline import Pipeline
from llm_server.core.implementations import ImageProcessor, ModelProcessor
from llm_server.core.types import MediaType, PipelineData

# Create a multi-step image processing pipeline
image_pipeline = Pipeline([
    ImageProcessor(max_size=(800, 800)),
    ModelProcessor(
        backend=your_model_backend,
        accepted_types=[MediaType.IMAGE],
        output_type=MediaType.TEXT
    )
])

# Execute with automatic validation and metrics
result = await image_pipeline.execute(initial_data)
```

### Circuit Breaker Integration

Protect your application from cascading failures:

```python
from llm_server.core.circuit_breaker import CircuitBreaker

@CircuitBreaker(failure_threshold=5, reset_timeout=60)
async def protected_model_call(input_data):
    # Your model call logic
    return await model.predict(input_data)
```

### Performance Monitoring

Track metrics across your application:

```python
from llm_server.core.metrics_wrappers import PerformanceMetrics, ModelBackendTracker

# Wrap your backends with performance tracking
metrics = PerformanceMetrics()
tracked_backend = ModelBackendTracker(your_backend, metrics)

# Get comprehensive performance data
performance_summary = metrics.get_summary()
```

## 🧪 Testing

### Run All Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run the full test suite
pytest tests/

# Run with coverage
pytest tests/ --cov=llm_server --cov-report=html
```

### Test Categories

- **Unit Tests**: Core protocol and implementation testing
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Circuit breaker and metrics validation

### Example Test

```python
@pytest.mark.anyio
async def test_custom_pipeline():
    """Test custom pipeline with multiple steps"""
    text_data = PipelineData(
        media_type=MediaType.TEXT, 
        content="test content", 
        metadata={}
    )
    
    pipeline = Pipeline([
        TextSummarizerStep(max_length=50),
        CustomProcessorStep()
    ])
    
    result = await pipeline.execute(text_data)
    assert result.metadata["summarized"] is True
```

## 📁 Project Structure

```
llm-server/
├── src/llm_server/           # Main application package
│   ├── core/                 # Core framework components
│   │   ├── protocols.py      # Interface definitions
│   │   ├── types.py          # Core data types
│   │   ├── pipeline.py       # Pipeline orchestration
│   │   ├── implementations.py # Standard implementations
│   │   ├── circuit_breaker.py # Reliability patterns
│   │   ├── metrics_*.py      # Performance monitoring
│   │   ├── opentelemetry_integration.py # OTel instruments
│   │   └── ...
│   ├── models/               # Model management
│   │   ├── manager.py        # Model lifecycle
│   │   ├── program_manager.py # DSPy program management
│   │   └── predictor.py      # Base prediction logic
│   └── defaults.py           # Default implementations
├── config/                   # Configuration files
│   └── model_config.yml      # Model definitions
├── tests/                    # Test suite
└── README.md                 # This file
```

## 🔍 Monitoring & Observability

### Metrics Available

- Request latency and throughput
- Token usage and cost tracking
- Circuit breaker state and recovery
- Pipeline step performance
- Model provider health

### Logging Features

- Structured JSON logging
- Distributed tracing with correlation IDs
- Performance metrics integration
- Error context preservation

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Run tests**: `pytest tests/`
4. **Run linting**: `ruff format . && ruff check .`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines

- Follow the protocol-based architecture patterns
- Add comprehensive tests for new features
- Update documentation for API changes
- Use type hints and maintain type safety
- Follow the existing code style (Ruff configuration)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🆘 Support

- **Documentation**: Check the inline code documentation
- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

---

**Built with** ❤️ **using FastAPI, DSPy, and modern Python patterns**