from collections.abc import Callable, Coroutine
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from a_simple_llm_kit.core.config import FrameworkSettings

if TYPE_CHECKING:
    from opentelemetry.trace import StatusCode
# ------------------------------------------------------------------------------
# 1. CONFIGURATION-DRIVEN SETUP
# ------------------------------------------------------------------------------
# Initialize settings to check if OTel should be enabled.
_settings = FrameworkSettings()

try:
    # This is the master switch. If disabled in config, we raise an error
    # to fall back to the dummy implementation immediately.
    if not _settings.otel_enabled:
        raise ImportError("OpenTelemetry is disabled by configuration.")

    # These imports will only succeed if the optional [opentelemetry]
    # dependencies are installed.
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import SpanKind

    _OTEL_ENABLED = True
    print("✅ OpenTelemetry integration is ENABLED.")

    # --------------------------------------------------------------------------
    # 2. RESOURCE DEFINITION (A Core OTel Best Practice)
    # --------------------------------------------------------------------------
    # This resource object attaches service metadata (e.g., name, version)
    # to every single metric and trace emitted by this library.
    _resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: _settings.otel_service_name,
            ResourceAttributes.SERVICE_VERSION: _settings.otel_service_version,
            "library.name": "a-simple-llm-kit-framework",
            "library.version": "0.2.0",  # This could be read from pyproject.toml
        }
    )

    # Note: The SDK (MeterProvider, TracerProvider) is configured by the
    # consuming application, which will automatically use this resource.

    # --------------------------------------------------------------------------
    # 3. GLOBAL METER AND TRACER
    # --------------------------------------------------------------------------
    # These are the entry points for creating all our instruments.
    _meter = metrics.get_meter("a-simple-llm-kit.framework")
    _tracer = trace.get_tracer("a-simple-llm-kit.framework")

    # ------------------------------------------------------------------------------
    # 5. METRIC INSTRUMENT DEFINITIONS
    # ------------------------------------------------------------------------------
    # All metrics are defined here, using the improved names and best practices.
    # They will be `None` if OTel is disabled.

    # Request-level metrics
    REQUESTS_TOTAL = (
        _meter.create_counter(
            name="a-simple-llm-kit.requests.total",
            description="Total number of requests processed by the framework.",
            unit="1",
        )
        if _OTEL_ENABLED
        else None
    )

    REQUEST_DURATION_SECONDS = (
        _meter.create_histogram(
            name="a-simple-llm-kit.request.duration_seconds",
            description="Histogram of request processing time in seconds.",
            unit="s",
        )
        if _OTEL_ENABLED
        else None
    )

    # Model-level metrics
    MODEL_CALLS_TOTAL = (
        _meter.create_counter(
            name="a-simple-llm-kit.model_calls.total",
            description="Total number of calls to a model backend.",
            unit="1",
        )
        if _OTEL_ENABLED
        else None
    )

    # Circuit Breaker metrics
    CIRCUIT_BREAKER_FAILURES_TOTAL = (
        _meter.create_counter(
            name="a-simple-llm-kit.circuit_breaker.failures_total",
            description="Total number of failures tracked by circuit breakers.",
            unit="1",
        )
        if _OTEL_ENABLED
        else None
    )

    CIRCUIT_BREAKER_STATE_CHANGES_TOTAL = (
        _meter.create_counter(
            name="a-simple-llm-kit.circuit_breaker.state_changes_total",
            description="Total number of times a circuit breaker's state has changed.",
            unit="1",
        )
        if _OTEL_ENABLED
        else None
    )

    CIRCUIT_BREAKER_STATE = (
        _meter.create_up_down_counter(
            name="a-simple-llm-kit.circuit_breaker.state",
            description="The current state of a circuit breaker (1 for OPEN, 0 otherwise).",
            unit="1",
        )
        if _OTEL_ENABLED
        else None
    )

    # Application & Business Metrics
    TOKEN_USAGE_TOTAL = (
        _meter.create_counter(
            name="app.token.usage.total",
            description="Total number of tokens processed, partitioned by type.",
            unit="1",
        )
        if _OTEL_ENABLED
        else None
    )

    TOKEN_COST_TOTAL = (
        _meter.create_counter(
            name="app.token.cost.total",
            description="Total estimated cost of token usage in USD.",
            unit="USD",
        )
        if _OTEL_ENABLED
        else None
    )

except ImportError:
    # --------------------------------------------------------------------------
    # 4. GRACEFUL DEGRADATION (Dummy Implementation)
    # --------------------------------------------------------------------------
    # If OTel is not installed or disabled, all OTel objects are set to None.
    # Code elsewhere can check `if _OTEL_ENABLED:` or `if REQUESTS_TOTAL:`
    # to safely skip instrumentation without crashing.
    print("⚠️ OpenTelemetry integration is DISABLED.")
    _OTEL_ENABLED = False
    _meter = _tracer = trace = Resource = SpanKind = None

    REQUESTS_TOTAL = None
    REQUEST_DURATION_SECONDS = None
    MODEL_CALLS_TOTAL = None
    CIRCUIT_BREAKER_FAILURES_TOTAL = None
    CIRCUIT_BREAKER_STATE_CHANGES_TOTAL = None
    CIRCUIT_BREAKER_STATE = None
    TOKEN_USAGE_TOTAL = None
    TOKEN_COST_TOTAL = None
#
# ------------------------------------------------------------------------------
# 6. TRACING HELPER UTILITIES
# ------------------------------------------------------------------------------
P = ParamSpec("P")
R = TypeVar("R")


def instrument_with_trace(
    name: str, span_kind: str = "INTERNAL"
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]:
    """
    Decorator to wrap an async function in an OpenTelemetry span.
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if (
                not _OTEL_ENABLED
                or not _tracer
                or not trace
                or not StatusCode
                or not SpanKind
            ):
                # If OTel is disabled, just call the original function directly.
                return await func(*args, **kwargs)

            try:
                otel_span_kind = SpanKind[span_kind.upper()]
            except KeyError:
                # Fallback to INTERNAL if the user provides an invalid string.
                otel_span_kind = SpanKind.INTERNAL

            # Use the refined nested `with` statement for clean context management
            with _tracer.start_as_current_span(name, kind=otel_span_kind) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR, description=str(e))
                    raise

        return wrapper

    return decorator
