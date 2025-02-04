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
image = modal.Image.from_dockerfile("Dockerfile.modal")

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
    """Single container running the FastAPI app with built-in health checks"""
    from fastapi import FastAPI
    from app.main import app as base_app
    from app.api.schemas.responses import HealthResponse
    import asyncio
    
    # Add internal healthcheck endpoint
    @base_app.get("/_internal/healthcheck", response_model=HealthResponse, include_in_schema=False)
    async def internal_healthcheck():
        return HealthResponse(status="healthy")

    # Schedule periodic health checks using FastAPI's background tasks
    @base_app.on_event("startup")
    async def schedule_healthchecks():
        async def run_periodic_healthcheck():
            while True:
                try:
                    await internal_healthcheck()
                    print("Internal health check passed!")
                except Exception as e:
                    print(f"Health check failed: {str(e)}")
                await asyncio.sleep(1800)  # 30 minutes

        asyncio.create_task(run_periodic_healthcheck())

    return base_app

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print(f"Starting {APP_NAME} in {ENV_NAME} environment")
    # The app will be served automatically by Modal
    pass

if __name__ == "__main__":
    app.run()