from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import OrderedDict, deque


def hash_job_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def valid_job_token(token: str, expected_hash: str) -> bool:
    return bool(token) and hmac.compare_digest(hash_job_token(token), expected_hash)


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 3600):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = time.monotonic()
            threshold = now - self.window_seconds
            # Last accepted timestamps are ordered, so expired clients can be
            # reclaimed without scanning every active client on every request.
            while self._events:
                oldest_events = next(iter(self._events.values()))
                if oldest_events and oldest_events[-1] >= threshold:
                    break
                self._events.popitem(last=False)
            events = self._events.setdefault(key, deque())
            while events and events[0] < threshold:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            self._events.move_to_end(key)
            return True
