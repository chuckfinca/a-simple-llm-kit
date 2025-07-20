import os
import modal

# Define the app name and environment
APP_NAME = "llm-server"
ENV_NAME = os.environ.get("APP_ENV", "development") # Use os.environ.get for safety

# Create a Modal Stub. This is the new name for modal.App.
# It acts as the main entrypoint for defining your Modal application.
stub = modal.Stub(f"{APP_NAME}-{ENV_NAME}")

# Create secrets object from an existing secret in your Modal account
app_secrets = modal.Secret.from_name("app-secrets")

# Create a persistent volume for logs
volume = modal.Volume.from_name(
    f"llm-server-{ENV_NAME}-logs", create_if_missing=True
)

# Define the container image
# This is the modern, declarative way to build the image.
# We start with a base, install requirements, and then copy the local code.
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("requirements.txt")
    .copy_local_dir(".", remote_path="/app")
    .run_commands("cd /app && pip install -e .") # Install the local package
    .workdir("/app") # Set the working directory
)

@stub.function(
    image=image,
    secrets=[app_secrets],
    volumes={"/data/logs": volume}, # Correctly mount the volume at the target path
    gpu="T4",
    memory=4096,
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    """This function defines the container's entrypoint."""
    from llm_server.main import app
    return app

# The if __name__ == "__main__": block is no longer needed for `modal serve`.
# To run this locally for development, you would now use the command:
# modal serve deploy_modal_app.py
# 