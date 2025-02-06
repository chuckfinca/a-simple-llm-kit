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

# Create image with requirements file and install packages
root = Path(__file__).parent
requirements_path = root / "requirements.txt"

image = (
    modal.Image.debian_slim(python_version="3.9")
    .add_local_dir(
        root,
        "/root/llm-server",  # Using more specific path
        exclude=[
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
    .run_commands(
        "cd /root/llm-server",  # Change to project directory
        "pip install -r requirements.txt",
        "pip install ."
    )
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
    import sys
    sys.path.append("/root")  # Add the root directory to Python path
    from app.main import app
    return app

if __name__ == "__main__":
    app.serve()