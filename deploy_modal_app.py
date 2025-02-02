import modal
import os
from typing import Optional

class ModalConfig:
    def __init__(self, base_name: str = "llm-server"):
        self.base_name = base_name
        self.app = modal.App(self.base_name)
        self.app_secrets = modal.Secret.from_name("app-secrets")
        self._volume = None
        self._image = None
    
    @property
    def env_name(self) -> str:
        """Get environment name from env vars or default to development"""
        return os.getenv("APP_ENV", "development")
    
    @property
    def app_name(self) -> str:
        """Get full application name including environment"""
        return f"{self.base_name}-{self.env_name}"
    
    @property
    def volume_name(self) -> str:
        """Get volume name for current environment"""
        return f"{self.base_name}-{self.env_name}-logs"
    
    @property
    def image(self) -> modal.Image:
        """Lazy load the Docker image"""
        if self._image is None:
            self._image = modal.Image.from_dockerfile("Dockerfile")
        return self._image
    
    @property
    def volume(self) -> modal.Volume:
        """Lazy load the volume"""
        if self._volume is None:
            self._volume = modal.Volume.from_name(
                self.volume_name, 
                create_if_missing=True
            )
        return self._volume
    
    def create_web_endpoint(self):
        """Create the FastAPI web endpoint"""
        @self.app.function(
            image=self.image,
            secrets=[self.app_secrets],
            volumes={"/data": self.volume},
            gpu="T4",
            memory=4096,
            timeout=600
        )
        @modal.asgi_app()
        def fastapi_app():
            from app.main import app
            return app
        
        return fastapi_app

    def create_healthcheck(self):
        """Create the healthcheck endpoint"""
        @self.app.function(
            image=self.image,
            secrets=[self.app_secrets],
            schedule=modal.Period(minutes=30)
        )
        def healthcheck():
            import requests
            response = requests.get("http://localhost:8000/health")
            assert response.status_code == 200
            print("Health check passed!")
            
        return healthcheck

# Create a global instance of ModalConfig
config = ModalConfig()

# Create and expose the app globally
app = config.app

# Initialize the endpoints
config.create_web_endpoint()
config.create_healthcheck()

if __name__ == "__main__":
    app.run()