from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable

from .models import SpeechRegion


class SpeechRegionIndex:
    """Sorted interval lookup that preserves the pipeline's existing region semantics."""

    def __init__(self, regions: Iterable[SpeechRegion]):
        self.regions = tuple(sorted(regions, key=lambda region: (region.start, region.end)))
        self._starts = tuple(region.start for region in self.regions)
        prefix_max_ends: list[float] = []
        maximum = float("-inf")
        for region in self.regions:
            maximum = max(maximum, region.end)
            prefix_max_ends.append(maximum)
        self._prefix_max_ends = tuple(prefix_max_ends)

    def overlapping(self, start: float, end: float) -> tuple[SpeechRegion, ...]:
        if end <= start or not self.regions:
            return ()
        left = bisect_right(self._prefix_max_ends, start)
        right = bisect_left(self._starts, end)
        return tuple(
            region
            for region in self.regions[left:right]
            if region.end > start and region.start < end
        )

    def first_containing(self, timestamp: float) -> tuple[int, SpeechRegion] | None:
        if not self.regions:
            return None
        left = bisect_left(self._prefix_max_ends, timestamp)
        right = bisect_right(self._starts, timestamp)
        for index in range(left, right):
            region = self.regions[index]
            if region.start <= timestamp <= region.end:
                return index, region
        return None

    def covered_seconds(self, start: float, end: float) -> float:
        return sum(
            min(end, region.end) - max(start, region.start)
            for region in self.overlapping(start, end)
        )
