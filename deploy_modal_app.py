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

# Use existing Dockerfile
image = modal.Image.from_dockerfile("Dockerfile")

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

@app.function(
    image=image,
    secrets=[app_secrets],
    schedule=modal.Period(minutes=30)
)
def healthcheck():
    import requests
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    print("Health check passed!")

if __name__ == "__main__":
    app.run()