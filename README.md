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

## Getting Started

You can run the server either using Docker for a fully containerized environment or directly on your local machine using `uv` for faster development iterations.

### Option 1: Running with Docker (Recommended for a Production-like Environment)

This method packages the entire application and its dependencies into a container, ensuring it runs the same way everywhere.

1.  **Set up your environment variables:** Create a `.env` file in the project root:
    ```bash
    cat > .env << EOL
    OPENAI_API_KEY=your_key_here
    ANTHROPIC_API_KEY=your_key_here
    HUGGINGFACE_API_KEY=your_key_here
    LLM_SERVER_API_KEY=your_server_key_here
    GEMINI_API_KEY=your_key_here
    EOL
    ```

2.  **Build and run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    The server will be available at `http://localhost:8000`.

### Option 2: Running Locally with `uv` (Recommended for Active Development)

This method is ideal for writing code, running tests, and using development tools, as it provides faster feedback and hot-reloading.

1.  **Install `uv`:** If you don't have it, follow the [official installation guide](https://github.com/astral-sh/uv).

2.  **Create the virtual environment:**
    ```bash
    uv venv
    ```

3.  **Install dependencies and development tools:**
    ```bash
    uv pip install -e ".[dev]"
    ```
    This command installs the server in "editable" mode and includes all tools like `pytest`, `ruff`, and `black`.

4.  **Run the application:**
    ```bash
    uv run llm-server
    ```
    For development with automatic hot-reloading, set the `LLM_SERVER_RELOAD` environment variable:
    ```bash
    LLM_SERVER_RELOAD=true uv run llm-server
    ```

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
      "prompt": "How many 'r's in 'strawberry'?",
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
      "prompt": "How many 'r's in 'strawberry'?",
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

This endpoint extracts structured contact information from an image of a business card.

**Request:**

```bash
# Cross-platform compatible way to encode an image as base64
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
        "model_id": "gpt-4o-mini"
      }
    }
  }'
```

**Example Success Response:**

The data field will contain a structured ExtractContact object.

```json
{
  "success": true,
  "data": {
    "name": {
      "prefix": null,
      "given_name": "John",
      "middle_name": null,
      "family_name": "Smith",
      "suffix": null
    },
    "work": {
      "job_title": "Software Engineer",
      "department": "Technology",
      "organization_name": "Innovate Corp"
    },
    "contact": {
      "phone_numbers": [
        {
          "label": "work",
          "value": "123-456-7890"
        }
      ],
      "email_addresses": [
        {
          "label": "work",
          "value": "john.smith@innovatecorp.com"
        }
      ],
      "postal_addresses": [
        {
          "label": "work",
          "value": {
            "street": "123 Innovation Drive",
            "city": "Techville",
            "state": "CA",
            "postal_code": "12345",
            "country": "USA"
          }
        }
      ],
      "url_addresses": [
        {
          "label": "work",
          "value": "www.innovatecorp.com"
        }
      ],
      "social_profiles": []
    },
    "notes": null
  },
  "error": null,
  "timestamp": "2025-07-23T22:00:00.123456Z",
  "metadata": {
    "program": {
      "id": "contact_extractor",
      "version": "1.0.0",
      "name": "Contact Extractor"
    },
    "model": {
      "id": "gpt-4o-mini",
      "provider": "openai",
      "base_model": "gpt-4o-mini",
      "model_name": "openai/gpt-4o-mini"
    },
    "performance": {
      "timing": {
        "total_ms": 1250.45
      },
      "tokens": {
        "input": 1245,
        "output": 387,
        "total": 1632,
        "cost_usd": 0.001547
      },
      "trace_id": "a7b3c9e1-f8d2-4e6a-9b1c-8d5f7e9a2c4b"
    },
    "execution_id": "3a7e9f12-d8e2-4b01-9861-4f3a8e72c5a3"
  }
}
```

## API Response Format

All API responses follow a consistent, standardized envelope to ensure predictability.

```json
{
  "success": true,
  "data": {
    // Response data specific to the endpoint, e.g., {"response": "The answer is..."}
  },
  "error": null,
  "timestamp": "2025-07-23T22:00:00.123456Z",
  "metadata": {
    "program": {
      "id": "text_completion",
      "version": "1.0.0",
      "name": "Predictor"
    },
    "model": {
      "id": "gpt-4o-mini",
      "provider": "openai",
      "base_model": "gpt-4o-mini",
      "model_name": "openai/gpt-4o-mini"
    },
    "performance": {
      "timing": {
        "total_ms": 750.25
      },
      "tokens": {
        "input": 50,
        "output": 150,
        "total": 200,
        "cost_usd": 0.0001
      },
      "trace_id": "a7b3c9e1-f8d2-4e6a-9b1c-8d5f7e9a2c4b"
    },
    "execution_id": "3a7e9f12-d8e2-4b01-9861-4f3a8e72c5a3",
    // Any additional request parameters (like temperature) will also appear here
    "temperature": 0.7
  }
}
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
from llm_server.core.types import MediaType, PipelineData
from llm_server.core.protocols import PipelineStep

class MyCustomStep(PipelineStep):
    @property
    def accepted_media_types(self) -> list[MediaType]:
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
from llm_server.core.pipeline import Pipeline

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
