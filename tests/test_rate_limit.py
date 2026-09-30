from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from llm_eval.rate_limit import ProviderRateLimiter, retry_delay


class FakeTime:
    def __init__(self, now=0.0):
        self.now = float(now)
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_kimi_waits_22_seconds_without_delaying_other_provider_limiter():
    fake = FakeTime()
    kimi = ProviderRateLimiter(22, clock=fake.clock, sleep_fn=fake.sleep)
    glm = ProviderRateLimiter(1, clock=fake.clock, sleep_fn=fake.sleep)

    assert kimi.wait() == 0
    assert kimi.wait() == pytest.approx(22)
    assert glm.wait() == 0


def test_retry_after_seconds_is_respected():
    error = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "7"}))
    assert retry_delay(error, attempt=1, now=100) == 7


def test_retry_after_http_date_is_respected():
    retry_at = format_datetime(datetime.fromtimestamp(130, tz=timezone.utc))
    error = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": retry_at}))
    assert retry_delay(error, attempt=1, now=100) == pytest.approx(30)


def test_epoch_reset_header_is_respected():
    error = SimpleNamespace(
        response=SimpleNamespace(headers={"X-RateLimit-Reset": "145"})
    )
    assert retry_delay(error, attempt=2, now=100) == pytest.approx(45)


def test_unparseable_headers_fall_back_to_exponential_backoff():
    error = SimpleNamespace(
        response=SimpleNamespace(
            headers={"Retry-After": "tomorrow", "X-RateLimit-Reset": "soon"}
        )
    )
    assert retry_delay(error, attempt=1, now=100) == 1
    assert retry_delay(error, attempt=3, now=100) == 4


def test_retry_delay_accepts_sanitized_attempt_headers():
    attempt = SimpleNamespace(retry_headers={"Retry-After": "9"})

    assert retry_delay(attempt, attempt=1, now=100) == 9
