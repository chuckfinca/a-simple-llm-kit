from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import Settings
from app.api.routes import router
from app.models.manager import ModelManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.model_manager = ModelManager(app.state.settings.config_path)
    yield
    app.state.model_manager = None

app = FastAPI(lifespan=lifespan)
app.include_router(router)