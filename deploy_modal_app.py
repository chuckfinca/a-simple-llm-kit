import modal
import os

# Create secrets
app_secrets = modal.Secret.from_name("app-secrets")

# Base app name
APP_NAME = "llm-server"

# Create the modal_app
app = modal.App(APP_NAME)

# Helper function to get environment name
def get_environment_name():
    """Get environment name from environment variable or default to development"""
    return os.getenv('APP_ENV', 'development')

# Create environment-specific configs
ENV_NAME = get_environment_name()
VOLUME_NAME = f"llm-server-{ENV_NAME}-logs"

# Create docker image with our package installed
image = modal.Image.from_dockerfile(
    "Dockerfile.modal",
    context_mount=modal.Mount.from_local_dir(".", remote_path="/app")
)

# Create volume for logs
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    secrets=[app_secrets],
    volumes={"/data": volume},
    gpu="T4",
    memory=4096,
    timeout=600,
    image = modal.Image.from_dockerfile(
        "Dockerfile.modal",
        context_mount=modal.Mount.from_local_dir(".", remote_path="/app")
    )
)

@modal.asgi_app()
def fastapi_app():
    from app.main import app
    return app

if __name__ == "__main__":
    app.serve()