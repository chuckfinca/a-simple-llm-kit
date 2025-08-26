from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from a_simple_llm_kit.core.metrics_wrappers import PerformanceMetrics

# Forward reference to avoid circular imports
_current_metrics: ContextVar[Optional["PerformanceMetrics"]] = ContextVar(
    "current_metrics", default=None
)


def get_current_metrics() -> Optional["PerformanceMetrics"]:
    return _current_metrics.get()


def set_current_metrics(metrics: Optional["PerformanceMetrics"]) -> None:
    """Sets the current metrics instance in the context."""
    _current_metrics.set(metrics)
