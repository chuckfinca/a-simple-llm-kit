import logging
import logging.config
import sys
from typing import Optional, ClassVar
import pydantic
from pathlib import Path

# Create default logger first
logger = logging.getLogger("llm_server")
logger.addHandler(logging.NullHandler())

class LogConfig(pydantic.BaseModel):
    """Logging configuration to be set for the server"""
    LOGGER_NAME: str = "llm_server"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"

    PROJECT_ROOT: ClassVar[Path] = Path(__file__).parent.parent.parent
    LOG_DIR: ClassVar[Path] = PROJECT_ROOT / "logs"

    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "format": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 10000000,  # 10MB
            "backupCount": 5,
        },
    }
    loggers: dict = {
        "llm_server": {"handlers": ["default", "file"], "level": LOG_LEVEL},
    }

def setup_logging(config: Optional[LogConfig] = None) -> None:
    """Configure logging for the application"""
    if config is None:
        config = LogConfig()
    
    # Ensure log directory exists
    config.LOG_DIR.mkdir(exist_ok=True)
    
    logging.config.dictConfig(config.model_dump())
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception

# Expose standard logging methods
def debug(msg: str, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)