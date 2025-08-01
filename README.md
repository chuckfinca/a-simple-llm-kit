LLM Server
A production-ready, extensible server for large language models with advanced pipeline processing, multi-modal capabilities, and enterprise-grade reliability features. Built with FastAPI, DSPy, and a composable architecture.
🚀 Key Features
Core Architecture

Pipeline-First Design: Composable, type-safe pipeline steps for complex processing workflows
Protocol-Based Framework: Clean interfaces enabling easy extension and testing
Multi-Modal Processing: Unified handling of text, images, and structured data
Performance Tracking: Comprehensive metrics collection with step-by-step timing analysis

Reliability & Monitoring

Circuit Breaker Pattern: Built-in failure protection with automatic recovery
Distributed Tracing: Request tracking across pipeline steps with unique trace IDs
Prometheus Integration: Production-ready metrics for monitoring and alerting
Structured Logging: JSON-formatted logs with context preservation

Model & Provider Support

Multi-Provider: OpenAI, Anthropic, Google Gemini, and Hugging Face
Flexible Configuration: YAML-based model configuration with parameter overrides
Token Management: Accurate token counting and cost estimation
Program Versioning: DSPy program management with optimization tracking

Specialized Capabilities

Contact Extraction: Advanced business card OCR with structured data output
Image Processing: Intelligent resizing, format conversion, and optimization
Type Safety: Full Pydantic integration with runtime protocol checking
Hot Reloading: Development-friendly auto-reload for rapid iteration

🏗️ Architecture Overview
The server is built around a composable pipeline architecture where each step implements the PipelineStep protocol:
python# Core pipeline concept
Pipeline([
    ImageProcessor(max_size=(800, 800)),           # Resize and optimize images
    ModelProcessor(backend, [MediaType.IMAGE]),    # Send to vision model
    OutputProcessor()                              # Format response
])
Key Components

Core Framework (src/llm_server/core/): Protocols, types, and base implementations
Model Management (src/llm_server/models/): Provider abstraction and program management
Pipeline System: Composable processing steps with automatic validation
Metrics & Monitoring: Performance tracking, circuit breakers, and observability

🚀 Quick Start
Option 1: Docker (Production-Ready)
bash# Set up environment variables
cat > .env << EOL
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
LLM_SERVER_API_KEY=your_server_key_here
EOL

# Build and run
docker-compose up --build
The server will be available at http://localhost:8000 with automatic metrics collection.
Option 2: Local Development with uv
bash# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up project
uv venv
uv pip install -e ".[dev]"

# Run with hot-reloading
LLM_SERVER_RELOAD=true uv run llm-server
📡 API Examples
Health & System Status
bashcurl -X GET http://localhost:8000/v1/health
Text Completion with Advanced Parameters
bashcurl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_server_key_here" \
  -d '{
    "request": {
      "prompt": "Explain quantum computing in simple terms",
      "model_id": "gpt-4o-mini",
      "temperature": 0.7,
      "top_p": 0.95,
      "frequency_penalty": 0.2,
      "presence_penalty": 0.1,
      "stop": [".", "\n"],
      "max_tokens": 500
    }
  }'
Business Card Contact Extraction
Extract structured contact information from business card images:
bash# Encode image as base64
IMAGE_B64=$(base64 < path/to/business_card.png | tr -d '\n')

curl -X POST http://localhost:8000/v1/extract-contact \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_server_key_here" \
  -d '{
    "request": {
      "pipeline_id": "extract-contact",
      "content": "'"$IMAGE_B64"'",
      "media_type": "image",
      "params": {
        "model_id": "gpt-4o-mini",
        "temperature": 0.1
      }
    }
  }'
Structured Response Example:
json{
  "success": true,
  "data": {
    "name": {
      "given_name": "John",
      "family_name": "Smith"
    },
    "work": {
      "job_title": "Software Engineer",
      "organization_name": "Innovate Corp"
    },
    "contact": {
      "phone_numbers": [{"label": "work", "value": "123-456-7890"}],
      "email_addresses": [{"label": "work", "value": "john.smith@innovatecorp.com"}],
      "postal_addresses": [{
        "label": "work",
        "value": {
          "street": "123 Innovation Drive",
          "city": "Techville",
          "state": "CA",
          "postal_code": "12345",
          "country": "USA"
        }
      }]
    }
  },
  "metadata": {
    "performance": {
      "timing": {"total_ms": 1250.45},
      "tokens": {"input": 1245, "output": 387, "cost_usd": 0.001547},
      "trace_id": "a7b3c9e1-f8d2-4e6a-9b1c-8d5f7e9a2c4b"
    },
    "model": {
      "id": "gpt-4o-mini",
      "provider": "openai"
    }
  }
}
📊 Response Format
All API responses follow a consistent envelope with comprehensive metadata:
json{
  "success": true,
  "data": {
    // Endpoint-specific response data
  },
  "error": null,
  "timestamp": "2025-08-01T10:30:00.123456Z",
  "metadata": {
    "program": {
      "id": "text_completion",
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
        "preparation_complete_ms": 12.34,
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
    "execution_id": "unique-execution-id"
  }
}
⚙️ Configuration
Model Configuration (config/model_config.yml)
yamlmodels:
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
Environment Variables
env# Required API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_KEY=your_hf_key
GEMINI_API_KEY=your_gemini_key

# Server Configuration
LLM_SERVER_API_KEY=your_secure_server_key
LLM_SERVER_RELOAD=false  # Set to true for development

# Optional: Custom config path
LLM_CONFIG_PATH=config/model_config.yml
🔧 Building Custom Pipelines
The framework's strength lies in its composable pipeline architecture. Create custom processing steps by implementing the PipelineStep protocol:
Custom Pipeline Step
pythonfrom llm_server.core.protocols import PipelineStep
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
Custom Model Backend
pythonfrom llm_server.core.protocols import ModelBackend
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
Combining Custom Components
pythonfrom llm_server.core.pipeline import Pipeline
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
🚢 Production Deployment
Modal.com (Recommended)
The project includes GitHub Actions workflows for seamless deployment:
yaml# .github/workflows/deploy_modal.yml
name: Deploy to Modal
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Modal
        run: |
          pip install modal
          modal deploy deploy_modal_app.py
Infrastructure Components

Cloudflare Tunnel: Secure ingress with automatic HTTPS
Prometheus Metrics: Application and system monitoring
Structured Logging: JSON logs with trace correlation
Health Checks: Automated service monitoring

See INFRASTRUCTURE.md for detailed deployment and monitoring setup.
🧪 Testing
Run All Tests
bash# Install test dependencies
uv pip install -e ".[dev]"

# Run the full test suite
pytest tests/

# Run with coverage
pytest tests/ --cov=llm_server --cov-report=html
Test Categories

Unit Tests: Core protocol and implementation testing
Integration Tests: End-to-end pipeline validation
Performance Tests: Circuit breaker and metrics validation

Example Test
python@pytest.mark.asyncio
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
📁 Project Structure
llm-server/
├── src/llm_server/           # Main application package
│   ├── core/                 # Core framework components
│   │   ├── protocols.py      # Interface definitions
│   │   ├── types.py          # Core data types
│   │   ├── pipeline.py       # Pipeline orchestration
│   │   ├── implementations.py # Standard implementations
│   │   ├── circuit_breaker.py # Reliability patterns
│   │   ├── metrics_*.py      # Performance monitoring
│   │   └── ...
│   ├── models/               # Model management
│   │   ├── manager.py        # Model lifecycle
│   │   ├── program_manager.py # DSPy program management
│   │   └── predictor.py      # Base prediction logic
│   └── defaults.py           # Default implementations
├── config/                   # Configuration files
│   └── model_config.yml      # Model definitions
├── tests/                    # Test suite
├── .github/workflows/        # CI/CD pipelines
├── INFRASTRUCTURE.md         # Deployment documentation
└── README.md                 # This file
🔍 Monitoring & Observability
Metrics Available

Request latency and throughput
Token usage and cost tracking
Circuit breaker state and recovery
Pipeline step performance
Model provider health

Logging Features

Structured JSON logging
Distributed tracing with correlation IDs
Performance metrics integration
Error context preservation

Health Endpoints
bash# Basic health check
curl http://localhost:8000/v1/health

# Detailed system status
curl http://localhost:8000/v1/health/detailed
🤝 Contributing

Fork the repository
Create a feature branch: git checkout -b feature/amazing-feature
Run tests: pytest tests/
Run linting: ruff format . && ruff check .
Commit changes: git commit -m 'Add amazing feature'
Push to branch: git push origin feature/amazing-feature
Open a Pull Request

Development Guidelines

Follow the protocol-based architecture patterns
Add comprehensive tests for new features
Update documentation for API changes
Use type hints and maintain type safety
Follow the existing code style (Ruff configuration)

📄 License
MIT License - see LICENSE file for details
🆘 Support

Documentation: Check the inline code documentation
Issues: Open GitHub issues for bugs and feature requests
Discussions: Use GitHub Discussions for questions and ideas


Built with ❤️ using FastAPI, DSPy, and modern Python patterns