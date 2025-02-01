from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core import logging
import traceback
import uuid

def _sanitize_request_data(request: Request) -> dict:
    return {
        'method': request.method,
        'path': request.url.path
    }

async def handle_exception(request: Request, exc: Exception) -> JSONResponse:
    error_id = str(uuid.uuid4())
    context = _sanitize_request_data(request)
    
    # Pass through the actual error information
    error_message = (
        getattr(exc, 'detail', str(exc))  # Use exc.detail if available (FastAPI), otherwise str(exc)
    )
    
    logging.error(
        f"Error {error_id}: {error_message}",
        extra={
            'error_id': error_id,
            **context,
            'error_type': exc.__class__.__name__,
            'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            'success': False,
            'error': error_message,
            'error_id': error_id
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    error_id = str(uuid.uuid4())
    context = _sanitize_request_data(request)
    
    sanitized_errors = [{
        'loc': err['loc'],
        'msg': err['msg']
    } for err in exc.errors()]
    
    logging.error(
        f"Validation error {error_id}",
        extra={
            'error_id': error_id,
            **context,
            'validation_errors': sanitized_errors
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            'success': False,
            'error': 'Invalid request data',
            'error_id': error_id,
            'details': sanitized_errors
        }
    )