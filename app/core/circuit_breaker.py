from enum import Enum
from datetime import datetime, timedelta
import asyncio
from functools import wraps
from app.core import logging

class State(Enum):
    CLOSED = "CLOSED"       # Everything is normal
    OPEN = "OPEN"          # Circuit is broken
    HALF_OPEN = "HALF_OPEN"  # Testing if it's safe to resume

class CircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 10, 
                 reset_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = State.CLOSED
        self.failures = 0
        self.last_failure_time = None
        self.lock = asyncio.Lock()
        self.protected_function_name = None  # Store the name of the function being protected

    def __call__(self, func):
        self.protected_function_name = func.__name__  # Capture function name
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.lock:
                if self.state == State.OPEN:
                    if await self._should_reset():
                        logging.info(
                            f"Circuit breaker for '{self.protected_function_name}' attempting reset "
                            f"after {self.reset_timeout} seconds in OPEN state"
                        )
                        self.state = State.HALF_OPEN
                    else:
                        remaining_time = (
                            self.last_failure_time + 
                            timedelta(seconds=self.reset_timeout) - 
                            datetime.now()
                        )
                        logging.warning(
                            f"Circuit breaker for '{self.protected_function_name}' is OPEN. "
                            f"Blocking request. Will try reset in {remaining_time.seconds} seconds. "
                            f"Last failure was at {self.last_failure_time}"
                        )
                        raise RuntimeError(
                            f"Circuit breaker is OPEN for '{self.protected_function_name}'. "
                            f"Too many failures (threshold: {self.failure_threshold}). "
                            f"Retry after {remaining_time.seconds} seconds"
                        )

                try:
                    if self.state == State.HALF_OPEN:
                        logging.info(
                            f"Circuit breaker for '{self.protected_function_name}' is HALF-OPEN. "
                            f"Testing with single request..."
                        )
                    
                    result = await func(*args, **kwargs)
                    
                    if self.state == State.HALF_OPEN:
                        self.state = State.CLOSED
                        self.failures = 0
                        logging.info(
                            f"Circuit breaker for '{self.protected_function_name}' test succeeded. "
                            f"Resetting to CLOSED state."
                        )
                    return result
                
                except Exception as e:
                    await self._handle_failure(e)
                    raise

        return wrapper

    async def _should_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        reset_after = self.last_failure_time + timedelta(seconds=self.reset_timeout)
        return datetime.now() >= reset_after

    async def _handle_failure(self, exception: Exception):
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        if self.state == State.HALF_OPEN or self.failures >= self.failure_threshold:
            old_state = self.state
            self.state = State.OPEN
            
            # Log detailed failure information
            # Only log detailed message when circuit first opens
            logging.error(
                f"Circuit breaker opened for '{self.protected_function_name}' after {self.failures} "
                f"failures. Last error: {type(exception).__name__}: {str(exception)}. "
                f"Will reset in {self.reset_timeout}s"
            )
        else:
            # Log warning for accumulating failures
            # Only log every other failure to reduce noise
            if self.failures % 2 == 0:
                logging.warning(
                    f"Circuit breaker for '{self.protected_function_name}': "
                    f"{self.failures}/{self.failure_threshold} failures"
                )