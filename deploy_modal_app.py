import modal
import os
from pathlib import Path

def read_requirements(filename="requirements.txt"):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Create secrets
app_secrets = modal.Secret.from_name("app-secrets")

# Base app name
APP_NAME = "llm-server"
ENV_NAME = os.getenv('APP_ENV', 'development')
VOLUME_NAME = f"llm-server-{ENV_NAME}-logs"

# Read requirements
requirements = read_requirements()

# Create image with app's dependencies from requirements.txt
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(*requirements)  # Unpack the requirements list
    .copy_local_dir(".", remote_path="/app")
)

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