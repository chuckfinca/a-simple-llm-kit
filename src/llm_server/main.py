from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from llm_server.api.routes import health_router, main_router
from llm_server.core import logging
from llm_server.core.config import get_settings
from llm_server.core.metrics_middleware import add_metrics_middleware
from llm_server.models.manager import ModelManager

logging.setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifespan is now much simpler.
    app.state.settings = get_settings()
    app.state.model_manager = ModelManager(app.state.settings.config_path)
    yield # No more program_manager

app = FastAPI(lifespan=lifespan)

# CORS middleware is still needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics and health checks are still valuable
add_metrics_middleware(app)
app.include_router(health_router)

# The main router now only contains the /extract-contact endpoint
app.include_router(main_router)
