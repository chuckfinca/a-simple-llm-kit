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
image = (
    modal.Image.debian_slim(python_version="3.9")
    .copy_local_dir(".", remote_path="/app")
    .run_commands([
        "cd /app && pip install -r requirements.txt",
        "cd /app && pip install -e ."
    ])
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
    import os
    # Set the working directory to /app
    os.chdir('/app')
    from llm_server.main import app
    return app

if __name__ == "__main__":
    app.serve()