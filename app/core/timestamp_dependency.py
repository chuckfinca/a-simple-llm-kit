from fastapi import Header, HTTPException, Request
from typing import Optional
import time
from app.core import logging

async def validate_request_timestamp(
    request: Request,
    x_request_time: Optional[str] = Header(None, description="Client-side timestamp in milliseconds since epoch")
):
    """
    FastAPI dependency that validates the X-Request-Time header.
    Raises an HTTPException if the header is missing or invalid.
    
    Args:
        request: The FastAPI request object
        x_request_time: The X-Request-Time header value
        
    Returns:
        Validated timestamp as float (milliseconds since epoch)
    """
    if x_request_time is None:
        logging.warning(f"Missing X-Request-Time header for {request.url.path}")
        raise HTTPException(
            status_code=400,
            detail="X-Request-Time header is required. Use milliseconds since epoch (e.g., JavaScript's Date.now() or Python's int(time.time() * 1000))"
        )
    
    try:
        timestamp = float(x_request_time)
        
        # Validate timestamp is reasonable (not too old or in the future)
        current_time = time.time() * 1000
        
        # If timestamp is more than 1 hour old
        if timestamp < current_time - (3600 * 1000):
            logging.warning(f"X-Request-Time too old: {timestamp} for {request.url.path}")
            raise HTTPException(
                status_code=400,
                detail="X-Request-Time is too old"
            )
            
        # If timestamp is more than 10 seconds in the future
        if timestamp > current_time + (10 * 1000):
            logging.warning(f"X-Request-Time in future: {timestamp} for {request.url.path}")
            raise HTTPException(
                status_code=400,
                detail="X-Request-Time is in the future"
            )
        
        # Store the validated timestamp in request state for use by middleware
        request.state.client_timestamp = timestamp
        return timestamp
        
    except ValueError:
        logging.warning(f"Invalid X-Request-Time header: {x_request_time} for {request.url.path}")
        raise HTTPException(
            status_code=400,
            detail="X-Request-Time header must be milliseconds since epoch as a number (e.g., JavaScript's Date.now() or Python's int(time.time() * 1000))"
        )