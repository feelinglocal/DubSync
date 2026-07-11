from __future__ import annotations

import unicodedata

from dubsync.changes import flow_text_to_lines
from dubsync.models import Cue
from dubsync.style_profile import StyleProfile, derive_style_profile
from dubsync.tokenize import alphanumeric_signature, tokenize_cues
from dubsync.verify import lint_cues


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1 for char in text)


def test_cjk_text_tokenizes_to_character_level():
    cue = Cue(index=1, start_ms=0, end_ms=500, lines=["你好，世界"])

    tokens = tokenize_cues([cue])

    assert [token.normalized for token in tokens] == ["你", "好", "世", "界"]
    assert alphanumeric_signature("你好，世界") == ["你", "好", "世", "界"]


def test_full_width_lines_use_visual_width_for_profile_and_lint():
    ok = Cue(index=1, start_ms=0, end_ms=500, lines=["界" * 13])
    too_long = Cue(index=2, start_ms=500, end_ms=1000, lines=["界" * 14])

    profile = derive_style_profile([too_long])
    issues = lint_cues([ok, too_long], StyleProfile(max_chars_per_line=26))

    assert profile.max_chars_per_line == 28
    assert [issue.cue_id for issue in issues if issue.kind == "line_length"] == [2]


def test_cjk_reflow_respects_visual_line_width():
    lines = flow_text_to_lines("界" * 14, max_chars=26, max_lines=2)

    assert len(lines) == 2
    assert all(_display_width(line) <= 26 for line in lines)
