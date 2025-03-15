# LLM Server

A lightweight, extensible server for working with large language models, focused on pipeline processing and multi-modal capabilities. Built with FastAPI and DSPy.

## Key Features

- **Pipeline Architecture**: Process text and images through customizable steps
- **Multi-Model Support**: Use models from OpenAI, Anthropic, Hugging Face, and Gemini
- **Circuit Breaker Pattern**: Built-in failure protection for model APIs
- **Contact Extraction**: Extract structured data from business card images
- **Type Safety**: Full typing support with Pydantic and runtime protocol checking
- **Monitoring**: Prometheus integration for metrics
- **Flexible Parameters**: Supports additional model parameters for fine-tuned control

## Quick Start with Docker Compose

The recommended way to run the server locally is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/chuckfinca/llm-server.git
cd llm-server

# Set up your environment variables in .env file
cat > .env << EOL
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
LLM_SERVER_API_KEY=your_server_key_here
GEMINI_API_KEY=your_key_here
EOL

# Run with Docker Compose
docker-compose up
```

The server will be available at `http://localhost:8000` with Prometheus metrics at `http://localhost:9090`.

## API Examples

### Health Check

```bash
curl -X GET http://localhost:8000/v1/health
```

### Text Completion

Basic usage:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_server_key_here" \
  -d '{
    "request": {
      "prompt": "Explain quantum computing in simple terms",
      "model_id": "gpt-4o-mini",
      "temperature": 0.7
    }
  }'
```

With additional parameters:

```bash
curl -X POST http://localhost:8000/v1/predict \
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
      "stop": [".", "\n"]
    }
  }'
```

### Contact Extraction from Image

```bash
# Cross-platform compatible way to encode an image as base64
IMAGE_B64=$(base64 < path/to/business_card.png | tr -d '\n')

curl -X POST http://localhost:8000/v1/extract-contact \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_server_key_here" \
  -d '{
    "request": {
      "pipeline_id": "extract-contact",
      "content": "'$IMAGE_B64'",
      "media_type": "image",
      "params": {
        "model_id": "gpt-4o-mini"
      }
    }
  }'
```

### Pipeline Processing

```bash
curl -X POST http://localhost:8000/v1/pipeline/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_server_key_here" \
  -d '{
    "request": {
      "pipeline_id": "text-processing",
      "content": "Analyze the sentiment of this text.",
      "media_type": "text",
      "params": {
        "model_id": "gpt-4o-mini"
      }
    }
  }'
```

## API Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {
    // Response data specific to the endpoint
  },
  "metadata": {
    "program_id": "text_completion",
    "program_version": "1.0.0",
    "program_name": "Predictor",
    "model_id": "gpt-4o-mini", 
    "model_info": {
      "provider": "openai",
      "base_model": "gpt-4o-mini",
      "model_name": "openai/gpt-4o-mini"
    },
    "request_id": "3a7e9f12-d8e2-4b01-9861-4f3a8e72c5a3",
    "timestamp": "2025-03-13T15:42:33.123456Z",
    // Any additional parameters provided will appear here
    "temperature": 0.7,
    "top_p": 0.95,
    "frequency_penalty": 0.2
  },
  "timestamp": "2025-03-13T15:42:33.123456Z"
}
```

## Manual Setup

If you prefer to run without Docker:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create a `.env` file or export directly)

3. Run the server:
```bash
python run.py
```

## Configuration

Models are configured in `config/model_config.yml`:

```yaml
models:
  gpt-4o-mini:
    model_name: "openai/gpt-4o-mini"
    max_tokens: 3000
  Meta-Llama-3.1-8B-Instruct:
    model_name: "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_tokens: 3000
  gemini-2.0-flash:
    model_name: "gemini/gemini-2.0-flash"
    max_tokens: 2048
```

## Deployment Options

The project includes GitHub Actions workflows for deploying to:

- Modal.com (recommended): `deploy_modal.yml`
- AWS: `deploy_aws.yml`

Refer to `INFRASTRUCTURE.md` for more details on deployment and infrastructure setup.

## Creating Custom Pipelines

The server uses a composable pipeline architecture:

```python
from app.core.types import MediaType, PipelineData
from app.core.protocols import PipelineStep

class MyCustomStep(PipelineStep):
    @property
    def accepted_media_types(self) -> List[MediaType]:
        return [MediaType.TEXT]
        
    async def process(self, data: PipelineData) -> PipelineData:
        # Process the data
        processed_content = do_something(data.content)
        
        return PipelineData(
            media_type=MediaType.TEXT,
            content=processed_content,
            metadata={**data.metadata, "processed": True}
        )
```

Then use it in your pipeline:

```python
from app.core.pipeline import Pipeline

pipeline = Pipeline([
    MyCustomStep(),
    ModelProcessor(backend, [MediaType.TEXT], MediaType.TEXT)
])

result = await pipeline.execute(initial_data)
```

## Testing

Run tests with:
```bash
pytest tests/
```

## Project Structure

```
.
├── app/
│   ├── api/            # FastAPI routes and schemas
│   ├── core/           # Core implementations
│   ├── models/         # Model interfaces
│   └── services/       # Business logic
├── config/             # Configuration files
├── .github/actions/    # Custom GitHub Actions
├── prometheus/         # Prometheus configuration
├── tests/              # Unit tests
├── docker-compose.yml  # Docker Compose configuration
└── run.py              # Server entry point
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License