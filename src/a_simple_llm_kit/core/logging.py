import json
import logging
import logging.config
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from a_simple_llm_kit.core.context import get_current_metrics

# Create default logger first
logger = logging.getLogger("a-simple-llm-kit")
logger.addHandler(logging.NullHandler())


class ContextualFilter(logging.Filter):
    """
    A logging filter that injects contextual information, like a trace_id,
    from a context variable into the log record.
    """

    def filter(self, record):
        # Attempt to get the current metrics object from the context
        metrics = get_current_metrics()
        if metrics:
            # If we found it, add its trace_id to the log record
            record.trace_id = metrics.trace_id
        else:
            # Otherwise, set a default value
            record.trace_id = "not-in-request"
        return True


class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON object, including any 'extra' data.
    """

    # Define the standard attributes of a LogRecord so we can exclude them
    # from the 'extra' data.
    RESERVED_ATTRS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record):
        # Start with the standard, required log fields
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Now, iterate through the record's dictionary and add any key that is
        # NOT a standard, reserved logging attribute. This correctly captures
        # all data passed in the 'extra' dictionary.
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS:
                log_object[key] = value

        # Use a safe default for JSON serialization to prevent crashes
        return json.dumps(log_object, default=str)


class LogConfig(BaseModel):
    """Logging configuration to be set for the server"""

    LOGGER_NAME: str = "a-simple-llm-kit"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"

    PROJECT_ROOT: ClassVar[Path] = Path(__file__).parent.parent.parent.parent
    LOG_DIR: ClassVar[Path] = PROJECT_ROOT / "logs"

    version: int = 1
    disable_existing_loggers: bool = False
    filters: dict = {
        "contextual": {
            "()": ContextualFilter,
        },
    }

    formatters: dict = {
        "default": {
            "()": JsonFormatter,
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "filters": ["contextual"],  # <-- 4. Attach the filter
        },
        "file": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 10000000,
            "backupCount": 5,
            "filters": ["contextual"],  # <-- 4. Attach the filter
        },
    }
    loggers: dict = {
        "a-simple-llm-kit": {"handlers": ["default", "file"], "level": LOG_LEVEL},
    }


def setup_logging(config: LogConfig | None = None) -> None:
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
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


# Expose standard logging methods
def debug(msg: str, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, exc_info=True, **kwargs):
    logger.error(msg, *args, exc_info=exc_info, **kwargs)


def critical(msg: str, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)
