import modal
import os
from pathlib import Path

# Get environment from env var, defaulting to staging
ENVIRONMENT = os.getenv("ENVIRONMENT", "staging")
APP_NAME = f"llm-server-{ENVIRONMENT}"

# Create the modal_app
modal_app = modal.App(APP_NAME)

# Create image from Dockerfile
image = modal.Image.add_local_dir("Dockerfile")

# Create volume for logs
volume = modal.Volume.from_name(f"{APP_NAME}-logs")

# Define the web endpoint function
@modal_app.function(
    image=image,
    secret=[
        modal.Secret.from_name(f"llm-server-{ENVIRONMENT}-secrets")
    ],
    volume=volume,
    gpu="T4",
    memory=4096,
    timeout=600
)
@modal.asgi_app()
def fastapi_app():
    # Import here to ensure imports happen inside container
    from app.main import app
    return app

# Create a healthcheck function
@modal_app.function(
    image=image,
    schedule=modal.Period(minutes=30)
)
def healthcheck():
    import requests
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    print("Health check passed!")

# Mount directories for persistent storage
MOUNT_POINTS = [
    ("/app/logs", volume),
]

for path, vol in MOUNT_POINTS:
    Path(path).mkdir(parents=True, exist_ok=True)
    
if __name__ == "__main__":
    modal_app.run()