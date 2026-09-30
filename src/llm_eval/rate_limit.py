"""Per-provider pacing and retry delay parsing."""

from __future__ import annotations

import threading
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping


class ProviderRateLimiter:
    """Thread-safe minimum interval limiter for one provider queue."""

    def __init__(
        self,
        min_interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("minimum interval cannot be negative")
        self.min_interval_seconds = float(min_interval_seconds)
        self.clock = clock
        self.sleep_fn = sleep_fn
        self._last_started: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> float:
        with self._lock:
            now = self.clock()
            delay = 0.0
            if self._last_started is not None:
                delay = max(
                    0.0,
                    self._last_started + self.min_interval_seconds - now,
                )
            if delay:
                self.sleep_fn(delay)
            self._last_started = self.clock()
            return delay


def _headers(error: Any) -> Mapping[str, str]:
    headers = getattr(error, "retry_headers", None)
    response = getattr(error, "response", None)
    if headers is None:
        headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(error, "headers", None)
    if not isinstance(headers, Mapping):
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}


def retry_delay(error: Any, attempt: int, now: float) -> float:
    """Parse common throttling headers, then fall back to bounded backoff."""

    headers = _headers(error)
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(retry_after)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, parsed.timestamp() - now)
            except (TypeError, ValueError, OverflowError):
                pass

    reset = headers.get("x-ratelimit-reset")
    if reset:
        try:
            return max(0.0, float(reset) - now)
        except ValueError:
            pass

    safe_attempt = max(1, int(attempt))
    return min(60.0, float(2 ** (safe_attempt - 1)))
