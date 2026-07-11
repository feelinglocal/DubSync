from __future__ import annotations

import json

from dubsync.models import Cue
from dubsync.overlap_detection import FixtureOverlapDetectionAdapter, overlap_flags_for_regions


def test_fixture_overlap_detector_flags_cues_intersecting_regions(tmp_path):
    fixture_path = tmp_path / "overlap.json"
    fixture_path.write_text(
        json.dumps({"regions": [{"start": 0.25, "end": 0.75, "confidence": 0.88}]}),
        encoding="utf-8",
    )
    cues = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["hello"], speaker_id="A"),
        Cue(index=2, start_ms=700, end_ms=1200, lines=["there"], speaker_id="B"),
    ]

    regions = FixtureOverlapDetectionAdapter(fixture_path).detect(tmp_path / "episode.wav")
    flags = overlap_flags_for_regions(cues, regions)

    assert regions[0].start == 0.25
    assert flags[0].kind == "overlap_detected"
    assert flags[0].cue_ids == [1, 2]
    assert flags[0].confidence == 0.88
