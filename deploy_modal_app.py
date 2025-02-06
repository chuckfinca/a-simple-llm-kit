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
stub_dir = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.9")
    .add_local_file(stub_dir / "requirements.txt", "/root/requirements.txt")
    .run_commands(
        "pip install -r /root/requirements.txt",
        "pip install ."  # This will install the current directory as a package
    )
    .add_local_dir(
        stub_dir,
        "/root/app",
        exclude=[".*", "__pycache__", "*.pyc", "*.pyo", "*.pyd", "build", "dist"]
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