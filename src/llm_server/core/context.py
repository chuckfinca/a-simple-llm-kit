from contextvars import ContextVar
from typing import Optional

# Forward reference to avoid circular imports
_current_metrics: ContextVar[Optional['PerformanceMetrics']] = ContextVar('current_metrics', default=None)

def get_current_metrics() -> Optional['PerformanceMetrics']:
    return _current_metrics.get()