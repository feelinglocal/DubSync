from __future__ import annotations

import json

import pytest

from dubsync.llm_providers import (
    GeminiLLMAdapter,
    _adjudication_prompt,
    _episode_context_payload,
    _gemini_source_context,
)
from dubsync.models import Cue, DivergenceSpan


def _expand(serialized: str) -> list[dict[str, object]]:
    table = json.loads(serialized)
    assert table["format"] == "dubsync.source-context.v2"
    assert table["alias_index_base"] == 0
    assert table["missing_trailing_cells"] is None
    decoded = []
    for sequence, row in enumerate(table["rows"], 1):
        cells = dict(zip(table["columns"], row + [None] * (len(table["columns"]) - len(row))))
        decoded.append({
            "cue_id": cells["cue_id"], "sequence_position": sequence,
            "start_ms": cells["start_ms"], "end_ms": cells["end_ms"], "source_lines": cells["source_lines"],
            "speaker_id": None if cells["speaker_alias"] is None else table["aliases"]["speaker_alias"][cells["speaker_alias"]],
            "character": None if cells["character_alias"] is None else table["aliases"]["character_alias"][cells["character_alias"]],
        })
    return decoded


def _cues() -> list[Cue]:
    return [
        Cue(index=20, start_ms=1491800, end_ms=1492222, lines=['“Eu…”', "Outra linha"], speaker_id="speaker_0", character="José"),
        Cue(index=4, start_ms=1491950, end_ms=1492500, lines=["なに？", "", "[PORTA FECHA]"]),
        Cue(index=900, start_ms=1493000, end_ms=1493200, lines=["Não."], speaker_id="speaker_1", character="Ana"),
    ]


def test_compact_context_roundtrip_preserves_ids_source_order_overlap_and_linebreaks():
    cues = _cues()
    assert _expand(_gemini_source_context(cues)) == _episode_context_payload(cues)


def test_aliases_distinguish_unknown_empty_strings_and_literal_zero():
    identities = [(None, None), (None, ""), ("", ""), ("0", "0"), ("speaker_b", "Hero"), ("speaker_b", "Hero")]
    cues = [Cue(index=i, start_ms=i*100, end_ms=i*100+50, lines=["line"], speaker_id=speaker, character=character)
            for i, (speaker, character) in enumerate(identities, 1)]
    serialized = _gemini_source_context(cues)
    table = json.loads(serialized)
    assert table["aliases"]["speaker_alias"] == ["", "0", "speaker_b"]
    assert table["aliases"]["character_alias"] == ["", "0", "Hero"]
    assert len(table["rows"][0]) == 4
    assert table["rows"][1][4:] == [None, 0]
    assert table["rows"][2][4:] == [0, 0]
    assert table["rows"][4][4:] == table["rows"][5][4:]
    assert _expand(serialized) == _episode_context_payload(cues)


def test_empty_context_has_a_lossless_empty_table():
    assert _expand(_gemini_source_context([])) == []


@pytest.mark.parametrize("configure_audio_first", [False, True])
def test_only_owned_audio_cache_uses_compact_serialization(tmp_path, configure_audio_first):
    path = tmp_path / "episode.wav"
    path.write_bytes(b"local fixture")
    cues = _cues()
    adapter = GeminiLLMAdapter(api_key="unused")
    if configure_audio_first:
        adapter.set_audio_context(path, duration_seconds=60)
        adapter.set_episode_context(cues)
    else:
        adapter.set_episode_context(cues)
        adapter.set_audio_context(path, duration_seconds=60)
    assert _expand(adapter.audio_context.source_context) == _episode_context_payload(cues)
    span = DivergenceSpan(case_id="case", cue_ids=[20], srt_text="Eu", asr_text="Eu")
    ordinary_prompt = json.loads(_adjudication_prompt([span], episode_context=cues))
    assert ordinary_prompt["episode_context"] == _episode_context_payload(cues)
    cues[0].lines[0] = "caller changed source"
    assert _expand(adapter.audio_context.source_context)[0]["source_lines"][0] == '“Eu…”'
    adapter.close()


def test_compact_context_reduces_repeated_metadata_by_at_least_thirty_percent():
    cues = [Cue(index=i, start_ms=i*1000, end_ms=i*1000+900, lines=["Primeira linha.", "Segunda linha."],
                speaker_id="speaker_0", character="Personagem") for i in range(1, 101)]
    expanded = json.dumps(_episode_context_payload(cues), ensure_ascii=False).encode("utf-8")
    compact = _gemini_source_context(cues).encode("utf-8")
    assert len(compact) <= len(expanded) * .7
    assert _expand(compact.decode("utf-8")) == _episode_context_payload(cues)
