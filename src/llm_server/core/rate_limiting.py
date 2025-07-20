import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import NamedTuple, Optional

from fastapi import HTTPException, Request

from llm_server.core.config import get_settings


class RateLimit(NamedTuple):
    unauthenticated: int
    authenticated: int
    window: int = 60  # seconds


class RateLimiter(ABC):
    @abstractmethod
    async def check_rate_limit(self, key: str, limit: int, window: int) -> None:
        pass


class InMemoryRateLimiter(RateLimiter):
    def __init__(self):
        self.requests = defaultdict(list)

    async def check_rate_limit(self, key: str, limit: int, window: int) -> None:
        now = time.time()
        self.requests[key] = [
            req_time for req_time in self.requests[key] if now - req_time < window
        ]

        if len(self.requests[key]) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {limit} requests per {window} seconds",
            )

        self.requests[key].append(now)


_rate_limiter = InMemoryRateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


# Default limits
DEFAULT_LIMITS = RateLimit(unauthenticated=5, authenticated=1000)


def rate_limit(limits: Optional[RateLimit] = None):
    if limits is None:
        limits = DEFAULT_LIMITS

    async def rate_limit_dependency(request: Request) -> None:
        settings = get_settings()
        limiter = get_rate_limiter()

        # Check if request has valid API key
        api_key = request.headers.get("X-API-Key")
        is_authenticated = api_key == settings.llm_server_api_key

        # Apply appropriate limit
        limit = limits.authenticated if is_authenticated else limits.unauthenticated
        key = api_key or request.client.host

        await limiter.check_rate_limit(key, limit, limits.window)

    return rate_limit_dependency
