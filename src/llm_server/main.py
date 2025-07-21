from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware

from llm_server.api.routes import health_router, main_router
from llm_server.api.routes_programs import programs_router
from llm_server.core import logging
from llm_server.core.config import get_settings
from llm_server.core.error_handling import (
    handle_exception,
    validation_exception_handler,
)
from llm_server.core.metrics_middleware import add_metrics_middleware
from llm_server.core.middleware import add_versioning_middleware
from llm_server.core.orientation_debugger import setup_orientation_debugger
from llm_server.models.manager import ModelManager
from llm_server.models.program_manager import ProgramManager

logging.setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize settings and model manager
    app.state.settings = get_settings()
    app.state.model_manager = ModelManager(app.state.settings.config_path)

    # Initialize program manager for DSPy program tracking
    app.state.program_manager = ProgramManager(app.state.model_manager)
    logging.info("Program manager initialized")

    yield

    # Clean up
    app.state.model_manager = None
    app.state.program_manager = None


app = FastAPI(lifespan=lifespan)
app.add_exception_handler(Exception, handle_exception)
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add versioning middleware to enforce program/model versioning
add_versioning_middleware(app)

# Add after other middleware
add_metrics_middleware(app)

# Setup orientation_debugger
setup_orientation_debugger(app)

# Include health check routes (no authentication)
app.include_router(health_router)

# Include main API routes (with authentication)
app.include_router(main_router)

# Nest the programs router under the main router
main_router.include_router(programs_router)
