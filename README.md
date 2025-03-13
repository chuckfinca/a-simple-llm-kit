# LLM Server

A lightweight, extensible server for working with large language models, focused on pipeline processing and multi-modal capabilities. Built with FastAPI and DSPy.

## Key Features

- **Pipeline Architecture**: Process text and images through customizable steps
- **Multi-Model Support**: Use models from OpenAI, Anthropic, Hugging Face, and Gemini
- **Circuit Breaker Pattern**: Built-in failure protection for model APIs
- **Contact Extraction**: Extract structured data from business card images
- **Type Safety**: Full typing support with Pydantic and runtime protocol checking
- **Monitoring**: Prometheus integration for metrics

## Quick Start with Docker Compose

The recommended way to run the server locally is using Docker Compose:

```bash|
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

```bash|
curl -X GET http://localhost:8000/v1/health
```

### Text Completion

```bash|
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_server_key_here" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "model_id": "gpt-4o-mini",
    "temperature": 0.7
  }'
```

### Contact Extraction from Image

```bash|
curl -X POST http://localhost:8000/v1/extract-contact \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_server_key_here" \
  -d '{
    "pipeline_id": "extract-contact",
    "content": "'$(base64 -w 0 path/to/business_card.png)'",
    "media_type": "image",
    "params": {
      "model_id": "Qwen2-VL-7B-Instruct"
    }
  }'
```

### Pipeline Processing

```bash|
curl -X POST http://localhost:8000/v1/pipeline/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_server_key_here" \
  -d '{
    "pipeline_id": "text-processing",
    "content": "Analyze the sentiment of this text.",
    "media_type": "text",
    "params": {
      "model_id": "gpt-4o-mini"
    }
  }'
```

## Manual Setup

If you prefer to run without Docker:

1. Install dependencies:
```bash|
pip install -r requirements.txt
```

2. Set up environment variables (create a `.env` file or export directly)

3. Run the server:
```bash|
python run.py
```

## Configuration

Models are configured in `config/model_config.yml`:

```yaml|
models:
  gpt-4o-mini:
    model_name: "openai/gpt-4o-mini"
    max_tokens: 3000
  Meta-Llama-3.1-8B-Instruct:
    model_name: "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_tokens: 3000
  Qwen2-VL-7B-Instruct:
    model_name: "huggingface/Qwen/Qwen2-VL-7B-Instruct"
    max_tokens: 2538
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

```python|
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

```python|
from app.core.pipeline import Pipeline

pipeline = Pipeline([
    MyCustomStep(),
    ModelProcessor(backend, [MediaType.TEXT], MediaType.TEXT)
])

result = await pipeline.execute(initial_data)
```

## Testing

Run tests with:
```bash|
pytest tests/
```

## Project Structure

```|
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