from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque


def hash_job_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def valid_job_token(token: str, expected_hash: str) -> bool:
    return bool(token) and hmac.compare_digest(hash_job_token(token), expected_hash)


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 3600):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        threshold = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] < threshold:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True
