import os

from modal import App, Image, Secret, Volume, asgi_app

# Create secrets
app_secrets = Secret.from_name("app-secrets")

# Base app name
APP_NAME = "llm-server"

# Create the modal_app
app = App(APP_NAME)

ENV_NAME = os.getenv("APP_ENV", "development")
VOLUME_NAME = f"llm-server-{ENV_NAME}-logs"

# Create image
image = (
    Image.debian_slim(python_version="3.9")
    .add_local_dir(".", remote_path="/app")
    .run_commands(
        ["cd /app && pip install -r requirements.txt", "cd /app && pip install -e ."]
    )
)

# Create volume for logs
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    secrets=[app_secrets],
    volumes={"/data": volume},
    gpu="T4",
    memory=4096,
    timeout=600,
)
@asgi_app()
def fastapi_app():
    import os

    os.chdir("/app")
    from llm_server.main import app as main_fastapi_app

    return main_fastapi_app
