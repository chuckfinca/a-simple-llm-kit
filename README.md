# LLM Server

A lightweight, extensible server for working with large language models, focused on pipeline processing and multi-modal capabilities. Built with FastAPI and DSPy.

## Key Features

- **Pipeline Architecture**: Process text and images through customizable steps
- **Multi-Model Support**: Use models from OpenAI, Anthropic, or Hugging Face
- **Contact Extraction**: Extract structured data from business card images
- **Type Safety**: Full typing support with Pydantic and runtime protocol checking

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
```bash
# .env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

3. Run the server:
```bash
python run.py
```

The server will start at `http://localhost:8000`

## Usage Examples

### Text Completion

```python
import requests

response = requests.post("http://localhost:8000/predict", 
    json={
        "prompt": "Explain quantum computing",
        "model_id": "gpt-4o-mini",
        "temperature": 0.7
    }
)
print(response.json()["response"])
```

### Contact Extraction

```python
import requests
import base64

# Read image file
with open("business_card.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/extract-contact",
    json={
        "pipeline_id": "extract-contact",
        "content": image_data,
        "media_type": "image",
        "params": {
            "model_id": "Qwen2-VL-7B-Instruct"
        }
    }
)
print(response.json()["content"])
```

## Configuration

Models are configured in `config/model_config.yml`:

```yaml
models:
  gpt-4o-mini:
    model_name: "openai/gpt-4o-mini"
    max_tokens: 3000
  Qwen2-VL-7B-Instruct:
    model_name: "huggingface/Qwen/Qwen2-VL-7B-Instruct"
    max_tokens: 2538
```

## Creating Custom Pipelines

The server uses a composable pipeline architecture. Here's how to create a custom pipeline:

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
│   ├── api/          # FastAPI routes and schemas
│   ├── core/         # Core implementations
│   ├── models/       # Model interfaces
│   └── services/     # Business logic
├── config/           # Configuration files
├── tests/           
└── run.py           # Server entry point
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License