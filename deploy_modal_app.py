import modal
import os
from pathlib import Path

# Create secrets
app_secrets = modal.Secret.from_name("app-secrets")

# Base app name
APP_NAME = "llm-server"

# Create the modal_app
app = modal.App(APP_NAME)

ENV_NAME = os.getenv('APP_ENV', 'development')
VOLUME_NAME = f"llm-server-{ENV_NAME}-logs"

# Create image with local Python package
image = (
    modal.Image.debian_slim(python_version="3.9")
    .add_local_python_source(
        "llm_server",  # root folder name
        ignore=[
            ".*",
            "__pycache__",
            "*.pyc",
            "*.pyo", 
            "*.pyd",
            "build",
            "dist",
            "*.egg-info",
            "logs",
            ".git",
            ".github",
            "tests"
        ]
    )
    .pip_install_from_requirements("requirements.txt")
)

# Create volume for logs
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    secrets=[app_secrets],
    volumes={"/data": volume},
    gpu="T4",
    memory=4096,
    timeout=600
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app
    return app

if __name__ == "__main__":
    app.serve()