import json

import pytest

from dubsync.pipeline import _source_cues_for_run


def test_legacy_source_decoding_is_recorded_and_survives_resume(tmp_path):
    source = tmp_path / "source.srt"
    original = "1\n00:00:00,000 --> 00:00:01,000\nCafé déjà vu.\n".encode("cp1252")
    source.write_bytes(original)
    cues, metadata = _source_cues_for_run(source, tmp_path, None)
    assert cues[0].text == "Café déjà vu."
    assert any(flag["kind"] == "source_encoding_converted" for flag in metadata["ingest_flags"])
    assert source.read_bytes() == original
    (tmp_path / "ingest.json").write_text(json.dumps({"cues": [cue.model_dump() for cue in cues], **metadata}), encoding="utf-8")
    resumed, resumed_metadata = _source_cues_for_run(source, tmp_path, "asr")
    assert resumed == cues
    assert resumed_metadata == metadata


def test_empty_source_cue_is_reported_without_merging_adjacent_dialogue(tmp_path):
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nFirst.\n\n"
                      "2\n00:00:01,000 --> 00:00:02,000\n\n"
                      "3\n00:00:02,000 --> 00:00:03,000\nLast.\n", encoding="utf-8")
    cues, metadata = _source_cues_for_run(source, tmp_path, None)
    assert [(cue.index, cue.text) for cue in cues] == [(1, "First."), (3, "Last.")]
    flag = next(flag for flag in metadata["ingest_flags"] if flag["kind"] == "source_empty_cues_ignored")
    assert flag["cue_ids"] == []  # No delivered cue exists to attach this episode-level notice to.
    assert "Source cue numbers: 2." in flag["message"]
    assert metadata["empty_source_cues"][0]["index"] == 2


def test_all_empty_source_fails_before_paid_work(tmp_path):
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no nonempty subtitle cues"):
        _source_cues_for_run(source, tmp_path, None)


@pytest.mark.parametrize("stage", ["asr", "align", "adjudicate", "rebuild", "verify"])
@pytest.mark.parametrize("all_empty", [False, True])
def test_legacy_empty_ingest_requires_restart_before_paid_work(tmp_path, stage, all_empty):
    from dubsync.models import Cue

    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["" if all_empty else "Hello."]),
            Cue(index=2, start_ms=1000, end_ms=2000, lines=[""])]
    artifact = tmp_path / "ingest.json"
    original = json.dumps({"cues": [cue.model_dump() for cue in cues]})
    artifact.write_text(original, encoding="utf-8")
    # Source need not be read and saved ownership must not be rewritten in place.
    with pytest.raises(ValueError, match="resume from ingest"):
        _source_cues_for_run(tmp_path / "source.srt", tmp_path, stage)
    assert artifact.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
def test_bom_unicode_source_is_not_reported_as_a_legacy_encoding(tmp_path, encoding):
    # A BOM identifies UTF-16/32 losslessly, so there are no accented characters to check.
    source = tmp_path / "source.srt"
    source.write_bytes(("\ufeff1\r\n00:00:00,000 --> 00:00:01,000\r\nCafé déjà vu.\r\n").encode(encoding))
    cues, metadata = _source_cues_for_run(source, tmp_path, None)
    assert cues[0].text == "Café déjà vu."
    assert "ingest_flags" not in metadata

