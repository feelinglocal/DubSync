from __future__ import annotations

import pytest

from dubsync.models import SpeechRegion
from dubsync.region_index import SpeechRegionIndex


def test_region_index_matches_naive_overlap_and_coverage_queries():
    regions = [
        SpeechRegion(start=4.0, end=5.0),
        SpeechRegion(start=0.0, end=1.5),
        SpeechRegion(start=1.0, end=3.0),
        SpeechRegion(start=1.25, end=1.75),
        SpeechRegion(start=6.0, end=8.0),
    ]
    ordered = sorted(regions, key=lambda region: (region.start, region.end))
    index = SpeechRegionIndex(regions)

    for start, end in [(-1.0, 0.0), (0.0, 1.0), (1.5, 4.0), (1.25, 1.75), (4.5, 7.0), (9.0, 10.0)]:
        expected = [
            region
            for region in ordered
            if region.end > start and region.start < end
        ]
        expected_coverage = sum(
            max(0.0, min(end, region.end) - max(start, region.start))
            for region in ordered
        )

        assert list(index.overlapping(start, end)) == expected
        assert index.covered_seconds(start, end) == pytest.approx(expected_coverage)


def test_region_index_matches_naive_first_containing_region_at_inclusive_boundaries():
    regions = [
        SpeechRegion(start=2.0, end=4.0),
        SpeechRegion(start=0.0, end=2.0),
        SpeechRegion(start=1.0, end=3.0),
    ]
    ordered = sorted(regions, key=lambda region: (region.start, region.end))
    index = SpeechRegionIndex(regions)

    for timestamp in [-0.1, 0.0, 1.5, 2.0, 3.5, 4.0, 4.1]:
        expected = next(
            (region for region in ordered if region.start <= timestamp <= region.end),
            None,
        )
        match = index.first_containing(timestamp)

        assert (match[1] if match is not None else None) == expected
        if match is not None:
            assert index.regions[match[0]] == expected
