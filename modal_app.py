import modal

# Create a single app instance
app = modal.App()

# Use existing Dockerfile
image = modal.Image.from_dockerfile("Dockerfile")

# Create volume for logs - create_if_missing lets us handle first deploy
volume = modal.Volume.from_name("logs", create_if_missing=True)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("llm-server-secrets")],
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
    schedule=modal.Period(minutes=30)
)
def healthcheck():
    import requests
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    print("Health check passed!")