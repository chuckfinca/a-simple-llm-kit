import modal
import sys

def get_environment_from_name(app_name: str) -> str:
    """Extract environment from app name (llm-server-{environment})"""
    if not app_name.startswith('llm-server-'):
        raise ValueError("App name must start with 'llm-server-'")
    return app_name.split('llm-server-')[1]

# Get the app name from command line args
try:
    name_index = sys.argv.index('--name')
    APP_NAME = sys.argv[name_index + 1]
    if not APP_NAME.startswith('llm-server-'):
        raise ValueError()
except (ValueError, IndexError):
    print("Error: Specify --name llm-server-{staging|production}")
    sys.exit(1)
ENVIRONMENT = get_environment_from_name(APP_NAME)

# Create the modal_app
app = modal.App(APP_NAME)

# Use existing Dockerfile
image = modal.Image.from_dockerfile("Dockerfile")

# Create volume for logs
volume = modal.Volume.from_name(f"{APP_NAME}-logs", create_if_missing=True)

# Define the web endpoint function
@app.function(
    image=image,
    secrets=[modal.Secret.from_name(f"llm-server-{ENVIRONMENT}-secrets")],
    volumes={"/data": volume},
    gpu="T4",
    memory=4096,
    timeout=600
)

@modal.asgi_app()
def fastapi_app():
    from app.main import app
    return app

# Create a healthcheck function
@app.function(
    image=image,
    schedule=modal.Period(minutes=30)
)
def healthcheck():
    import requests
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    print("Health check passed!")

if __name__ == "__main__":
    app.run()
