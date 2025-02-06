import modal
import os
from pathlib import Path

# Create secrets
app_secrets = modal.Secret.from_name("app-secrets")

# Base app name
APP_NAME = "llm-server"

# Create the modal_app
modal_app = modal.App(APP_NAME)

ENV_NAME = os.getenv('APP_ENV', 'development')
VOLUME_NAME = f"llm-server-{ENV_NAME}-logs"

# Create image with requirements file and install packages
requirements_path = Path(__file__).parent / "requirements.txt"
image = (
    modal.Image.debian_slim(python_version="3.9")
    .copy_local_file(requirements_path, remote_path="/root/requirements.txt")
    .copy_local_dir(".", remote_path="/root/llm-server")
    .run_commands(
        "echo '=== Directory Structure ===' && ls -la /root/llm-server",
        "echo '=== Python Path ===' && python -c 'import sys; print(\"\n\".join(sys.path))'",
        "echo '=== Current Directory ===' && pwd",
        "cd /root/llm-server",
        "pip install -r /root/requirements.txt",
        "pip install -e .",
        "echo '=== Installed Packages ===' && pip list"
        # Add the llm-server directory to PYTHONPATH
        "export PYTHONPATH=/root/llm-server:$PYTHONPATH",
        # Print debug information
        "echo 'Current directory contents:' && ls -la",
        "echo 'Python path:' && python -c 'import sys; print(sys.path)'"
    )
)

# Create volume for logs
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@modal_app.function(
    image=image,
    secrets=[app_secrets, modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"})],
    volumes={"/data": volume},
    gpu="T4",
    memory=4096,
    timeout=600
)
@modal.asgi_app()
def fastapi_app():
    modal.logger.info("Starting fastapi_app function")
    try:
        modal.logger.info("Attempting to import app.main...")
        from app.main import app
        modal.logger.info("Successfully imported app")
        return app
    except ImportError as e:
        modal.logger.error(f"Import error: {str(e)}")
        import sys
        modal.logger.error(f"Python path: {sys.path}")
        raise
    except Exception as e:
        modal.logger.error(f"Unexpected error: {str(e)}")
        raise



if __name__ == "__main__":
    modal_app.serve()