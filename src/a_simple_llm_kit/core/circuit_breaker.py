import asyncio
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from a_simple_llm_kit.core import logging
from a_simple_llm_kit.core.utils import get_utc_now

# --- OTel Integration ---
# Import the new metric instruments. They will be `None` if OTel is disabled.
try:
    from a_simple_llm_kit.core.opentelemetry_integration import (
        CIRCUIT_BREAKER_FAILURES_TOTAL,
        CIRCUIT_BREAKER_STATE,
        CIRCUIT_BREAKER_STATE_CHANGES_TOTAL,
    )
except ImportError:
    CIRCUIT_BREAKER_FAILURES_TOTAL = None
    CIRCUIT_BREAKER_STATE = None
    CIRCUIT_BREAKER_STATE_CHANGES_TOTAL = None
# --- End OTel Integration ---

# --- Define Type Variables for the decorator ---
P = ParamSpec("P")
R = TypeVar("R")


class State(Enum):
    CLOSED = "CLOSED"  # Everything is normal
    OPEN = "OPEN"  # Circuit is broken
    HALF_OPEN = "HALF_OPEN"  # Testing if it's safe to resume


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 10, reset_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = State.CLOSED
        self.failures = 0
        self.last_failure_time = None
        self.lock = asyncio.Lock()
        self.protected_function_name = (
            None  # Store the name of the function being protected
        )

        # Metrics tracking
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "state_changes": [],
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "consecutive_failures": 0,
            "max_consecutive_failures": 0,
            "current_state": State.CLOSED.value,
            "time_in_states": {
                State.CLOSED.value: 0,
                State.OPEN.value: 0,
                State.HALF_OPEN.value: 0,
            },
            "state_change_timestamps": {
                State.CLOSED.value: get_utc_now(),
                State.OPEN.value: None,
                State.HALF_OPEN.value: None,
            },
        }

    def __call__(
        self, func: Callable[P, Coroutine[Any, Any, R]]
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        self.protected_function_name = func.__name__  # Capture function name

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self.metrics["total_calls"] += 1

            async with self.lock:
                # Update time spent in current state
                current_time = get_utc_now()
                last_state_change = self.metrics["state_change_timestamps"][
                    self.state.value
                ]
                if last_state_change:
                    time_delta = (current_time - last_state_change).total_seconds()
                    self.metrics["time_in_states"][self.state.value] += time_delta
                    # Update the timestamp for the current state
                    self.metrics["state_change_timestamps"][self.state.value] = (
                        current_time
                    )

                if self.state == State.OPEN:
                    if await self._should_reset():
                        # Track state change
                        old_state = self.state
                        self.state = State.HALF_OPEN
                        await self._track_state_change(old_state, self.state)

                        self.metrics["recovery_attempts"] += 1

                        logging.info(
                            f"Circuit breaker for '{self.protected_function_name}' attempting reset "
                            f"after {self.reset_timeout} seconds in OPEN state"
                        )
                    else:
                        # Add a check to assure the type checker and for robustness
                        if self.last_failure_time:
                            remaining_time = (
                                self.last_failure_time
                                + timedelta(seconds=self.reset_timeout)
                                - datetime.now()
                            )
                        else:
                            # This case should not be reached in the current logic,
                            # but it's safe to have a fallback.
                            remaining_time = timedelta(seconds=self.reset_timeout)

                        logging.warning(
                            f"Circuit breaker for '{self.protected_function_name}' is OPEN. "
                            f"Blocking request. Will try reset in {remaining_time.seconds} seconds. "
                            f"Last failure was at {self.last_failure_time}"
                        )
                        # Track blocked request
                        self.metrics["blocked_requests"] = (
                            self.metrics.get("blocked_requests", 0) + 1
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

                    # Handle successful execution
                    self.metrics["successful_calls"] += 1

                    if self.state == State.HALF_OPEN:
                        # Track state change and successful recovery
                        old_state = self.state
                        self.state = State.CLOSED
                        await self._track_state_change(old_state, self.state)

                        self.failures = 0
                        self.metrics["consecutive_failures"] = 0
                        self.metrics["successful_recoveries"] += 1

                        logging.info(
                            f"Circuit breaker for '{self.protected_function_name}' test succeeded. "
                            f"Resetting to CLOSED state."
                        )
                    return result

                except Exception as e:
                    # Handle failure
                    self.metrics["failed_calls"] += 1
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

        # --- OTel Instrumentation ---
        if CIRCUIT_BREAKER_FAILURES_TOTAL:
            attributes = {"function.name": self.protected_function_name or "unknown"}
            CIRCUIT_BREAKER_FAILURES_TOTAL.add(1, attributes)

        # Update consecutive failures metric
        self.metrics["consecutive_failures"] += 1
        self.metrics["max_consecutive_failures"] = max(
            self.metrics["max_consecutive_failures"],
            self.metrics["consecutive_failures"],
        )

        if self.state == State.HALF_OPEN or self.failures >= self.failure_threshold:
            old_state = self.state
            self.state = State.OPEN

            # Track state change
            await self._track_state_change(old_state, self.state)

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

    async def _track_state_change(self, from_state: State, to_state: State):
        """Track a state transition for metrics purposes"""
        current_time = get_utc_now()
        # Record time spent in previous state
        if self.metrics["state_change_timestamps"][from_state.value]:
            time_delta = (
                current_time - self.metrics["state_change_timestamps"][from_state.value]
            ).total_seconds()
            self.metrics["time_in_states"][from_state.value] += time_delta

        # Update state change metrics
        self.metrics["state_changes"].append(
            {
                "from": from_state.value,
                "to": to_state.value,
                "timestamp": current_time.isoformat(),
                "failures": self.failures,
            }
        )

        # Update current state metrics
        self.metrics["current_state"] = to_state.value

        # Reset timestamp for the new state
        self.metrics["state_change_timestamps"][to_state.value] = current_time

        # --- OTel Instrumentation ---
        attributes = {
            "function.name": self.protected_function_name,
            "from": from_state.value,
            "to": to_state.value,
        }
        if CIRCUIT_BREAKER_STATE_CHANGES_TOTAL:
            CIRCUIT_BREAKER_STATE_CHANGES_TOTAL.add(1, attributes)

        if CIRCUIT_BREAKER_STATE:
            # Breaker is OPEN (tripped)
            if to_state == State.OPEN:
                CIRCUIT_BREAKER_STATE.add(1, attributes)
            # Breaker is returning to CLOSED from OPEN or HALF_OPEN
            elif from_state == State.OPEN or from_state == State.HALF_OPEN:
                CIRCUIT_BREAKER_STATE.add(-1, attributes)

        # Log state change
        logging.info(
            f"Circuit breaker for '{self.protected_function_name}' state changed: "
            f"{from_state.value} -> {to_state.value} (failures: {self.failures})"
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get current circuit breaker metrics"""
        # Update time spent in current state
        current_time = get_utc_now()
        last_state_change = self.metrics["state_change_timestamps"][self.state.value]
        if last_state_change:
            time_delta = (current_time - last_state_change).total_seconds()
            current_state_time = self.metrics["time_in_states"][self.state.value]
            self.metrics["time_in_states"][self.state.value] = (
                current_state_time + time_delta
            )
            # Update the timestamp
            self.metrics["state_change_timestamps"][self.state.value] = current_time

        # Return a copy of the metrics to prevent external modification
        return self.metrics.copy()
