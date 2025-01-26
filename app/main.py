from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.routes import router
from app.core.error_handling import handle_exception, validation_exception_handler
from app.models.manager import ModelManager
from app.core import logging
from fastapi.exceptions import RequestValidationError

logging.setup_logging() 

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    app.state.model_manager = ModelManager(app.state.settings.config_path)
    yield
    app.state.model_manager = None

app = FastAPI(lifespan=lifespan)
app.add_exception_handler(Exception, handle_exception)
app.add_exception_handler(RequestValidationError, validation_exception_handler)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)