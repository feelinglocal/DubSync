from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from dubsync.cli import app
from dubsync.models import Cue
from dubsync.punctuation import StaticPunctuationAdapter, apply_punctuation_pass
from dubsync.providers import ProviderError
from dubsync.srt_io import parse_srt_text


class RecordingPunctuationAdapter:
    def __init__(self):
        self.batches: list[list[int]] = []

    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        self.batches.append([cue.index for cue in cues])
        return {}


class FailingPunctuationAdapter:
    def punctuate(self, cues: list[Cue]) -> dict[int, str]:
        raise ProviderError("provider timed out")


def test_punctuation_pass_batches_cues_by_scene_gap():
    adapter = RecordingPunctuationAdapter()
    cues = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["hello"]),
        Cue(index=2, start_ms=1000, end_ms=1500, lines=["there"]),
        Cue(index=3, start_ms=7000, end_ms=7500, lines=["again"]),
    ]

    updated, flags = apply_punctuation_pass(cues, adapter, scene_gap_seconds=4.0)

    assert adapter.batches == [[1, 2], [3]]
    assert updated == cues
    assert flags == []


def test_punctuation_provider_failure_preserves_source_cues_with_qc_flag():
    cues = [
        Cue(index=1, start_ms=0, end_ms=500, lines=["hello"]),
        Cue(index=2, start_ms=1000, end_ms=1500, lines=["there"]),
    ]

    updated, flags = apply_punctuation_pass(
        cues,
        FailingPunctuationAdapter(),
        scene_gap_seconds=4.0,
    )

    assert updated == cues
    assert [flag.kind for flag in flags] == ["punctuation_provider_unavailable"]
    assert flags[0].cue_ids == [1, 2]


def test_punctuation_pass_caps_dense_scene_batches():
    adapter = RecordingPunctuationAdapter()
    cues = [
        Cue(index=index, start_ms=index * 1_000, end_ms=index * 1_000 + 500, lines=[f"cue {index}"])
        for index in range(1, 46)
    ]

    updated, flags = apply_punctuation_pass(cues, adapter, scene_gap_seconds=4.0)

    assert [len(batch) for batch in adapter.batches] == [40, 5]
    assert updated == cues
    assert flags == []


def test_punctuation_pass_packs_many_sparse_scenes_for_long_episode_scaling():
    cues = [
        Cue(
            index=index,
            start_ms=index * 10_000,
            end_ms=index * 10_000 + 1_000,
            lines=[f"Line {index}"],
        )
        for index in range(1, 82)
    ]
    adapter = RecordingPunctuationAdapter()

    updated, flags = apply_punctuation_pass(cues, adapter, scene_gap_seconds=4.0)

    assert [len(batch) for batch in adapter.batches] == [40, 40, 1]
    assert [cue_id for batch in adapter.batches for cue_id in batch] == list(range(1, 82))
    assert updated == cues
    assert flags == []


def test_packed_punctuation_batches_retain_explicit_scene_identity():
    scene_batches: list[list[tuple[int | None, int | None]]] = []

    class SceneRecordingAdapter(RecordingPunctuationAdapter):
        def punctuate(self, cues):
            scene_batches.append(
                [
                    (cue.prompt_scene_id, cue.prompt_scene_position)
                    for cue in cues
                ]
            )
            return super().punctuate(cues)

    cues = [
        Cue(
            index=index,
            start_ms=index * 10_000,
            end_ms=index * 10_000 + 1_000,
            lines=[f"Line {index}"],
        )
        for index in range(1, 42)
    ]

    apply_punctuation_pass(cues, SceneRecordingAdapter(), scene_gap_seconds=4.0)

    assert scene_batches == [
        [(index, 1) for index in range(1, 41)],
        [(41, 1)],
    ]


def test_punctuation_pass_preserves_valid_proposed_line_breaks():
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["hello", "there"])]
    adapter = StaticPunctuationAdapter({1: "Hello,\nthere."})

    updated, flags = apply_punctuation_pass(cues, adapter)

    assert flags == []
    assert updated[0].lines == ["Hello,", "there."]


def test_punctuation_pass_restores_source_line_breaks_when_model_flattens_them():
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["Alessia hör auf", "mit dem Quatsch"])]
    adapter = StaticPunctuationAdapter({1: "Alessia, hör auf mit dem Quatsch!"})

    updated, flags = apply_punctuation_pass(cues, adapter)

    assert flags == []
    assert updated[0].lines == ["Alessia, hör auf", "mit dem Quatsch!"]


def test_punctuation_pass_rejects_model_added_german_dialogue_quotes():
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["Wer bist du?"])]
    adapter = StaticPunctuationAdapter({1: "„Wer bist du?“"})

    updated, flags = apply_punctuation_pass(cues, adapter)

    assert updated == cues
    assert [flag.kind for flag in flags] == ["invalid_punctuation_change"]
    assert flags[0].severity == "error"
    assert "quotation mark signature changed" in flags[0].message


def test_punctuation_pass_anchors_quotes_to_source_when_words_are_unchanged():
    source_cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["„Hallo Welt“"])]
    rebuilt_cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["Hallo Welt"])]
    adapter = StaticPunctuationAdapter({1: "„Hallo Welt“"})

    updated, flags = apply_punctuation_pass(
        rebuilt_cues,
        adapter,
        source_cues=source_cues,
    )

    assert flags == []
    assert updated[0].text == "„Hallo Welt“"


def test_punctuation_pass_validates_word_unchanged_cue_against_source_quotes():
    source_cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["„Wer bist du?“"])]
    rebuilt_cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=['"Wer bist du?"'])]
    adapter = StaticPunctuationAdapter({1: '"Wer bist du?"'})

    updated, flags = apply_punctuation_pass(rebuilt_cues, adapter, source_cues=source_cues)

    assert updated[0].start_ms == rebuilt_cues[0].start_ms
    assert updated[0].end_ms == rebuilt_cues[0].end_ms
    assert updated[0].text == source_cues[0].text
    assert [flag.kind for flag in flags] == ["invalid_punctuation_change"]
    assert "quotation mark signature changed" in flags[0].message


def test_punctuation_pass_allows_legitimate_quotes_in_word_changed_cue():
    source_cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["Hallo."])]
    rebuilt_cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=['Er sagte "Hallo".'])]
    adapter = StaticPunctuationAdapter({1: 'Er sagte "Hallo!"'})

    updated, flags = apply_punctuation_pass(rebuilt_cues, adapter, source_cues=source_cues)

    assert flags == []
    assert updated[0].text == 'Er sagte "Hallo!"'


def test_punctuation_pass_anchors_quotes_to_changed_text_when_words_changed():
    source = [Cue(index=1, start_ms=0, end_ms=1000, lines=["Wer bist du?"])]
    rebuilt = [Cue(index=1, start_ms=10, end_ms=990, lines=["„Bleib hier?“"])]
    adapter = StaticPunctuationAdapter({1: "„Bleib hier!“"})

    updated, flags = apply_punctuation_pass(rebuilt, adapter, source_cues=source)

    assert flags == []
    assert updated[0].text == "„Bleib hier!“"


def test_punctuation_pass_reflows_an_overlong_model_line():
    cues = [Cue(index=1, start_ms=0, end_ms=2700, lines=["Hello this is the Dubsync", "Cloud test"])]
    adapter = StaticPunctuationAdapter({1: "Hello, this is the Dubsync Cloud test."})

    updated, flags = apply_punctuation_pass(
        cues,
        adapter,
        max_chars_per_line=26,
        max_lines_per_cue=2,
    )

    assert flags == []
    assert updated[0].lines == ["Hello, this is the Dubsync", "Cloud test."]


def test_punctuation_pass_reflows_excess_model_lines():
    cues = [Cue(index=1, start_ms=0, end_ms=2700, lines=["Hello this is the Dubsync", "Cloud test"])]
    adapter = StaticPunctuationAdapter({1: "Hello,\nthis is the Dubsync\nCloud test."})

    updated, flags = apply_punctuation_pass(
        cues,
        adapter,
        max_chars_per_line=26,
        max_lines_per_cue=2,
    )

    assert flags == []
    assert updated[0].lines == ["Hello, this is the Dubsync", "Cloud test."]


def test_punctuation_pass_keeps_restored_source_breaks_for_unchanged_cue_even_when_profile_is_narrow():
    source_cues = [Cue(index=1, start_ms=0, end_ms=2700, lines=["Drachen-Evolutionssystem", "besitze ich längst"])]
    rebuilt_cues = [Cue(index=1, start_ms=40, end_ms=2680, lines=["Drachen-Evolutionssystem besitze", "ich längst"])]
    adapter = StaticPunctuationAdapter({1: "Drachen-Evolutionssystem besitze ich längst."})

    updated, flags = apply_punctuation_pass(
        rebuilt_cues,
        adapter,
        max_chars_per_line=22,
        max_lines_per_cue=2,
        source_cues=source_cues,
    )

    assert flags == []
    assert updated[0].lines == ["Drachen-Evolutionssystem", "besitze ich längst."]


def test_punctuation_pass_does_not_text_reflow_source_cue_that_exceeds_line_limit():
    source = Cue(
        index=1,
        start_ms=0,
        end_ms=2_700,
        lines=["Alpha beta", "gamma delta", "epsilon zeta."],
    )
    adapter = StaticPunctuationAdapter({1: "Alpha beta, gamma delta, epsilon zeta."})

    updated, flags = apply_punctuation_pass(
        [source],
        adapter,
        max_chars_per_line=80,
        max_lines_per_cue=2,
        source_cues=[source],
    )

    assert updated[0].lines == ["Alpha beta,", "gamma delta,", "epsilon zeta."]
    assert [flag.kind for flag in flags] == ["punctuation_source_structure_preserved"]
    assert flags[0].severity == "warning"


def test_punctuation_pass_allows_generation_without_source_cues():
    cues = [Cue(index=1, start_ms=0, end_ms=1000, lines=["hallo welt"])]
    adapter = StaticPunctuationAdapter({1: "Hallo Welt."})

    updated, flags = apply_punctuation_pass(cues, adapter, source_cues=[])

    assert flags == []
    assert updated[0].text == "Hallo Welt."


def test_cli_sync_applies_fixture_punctuation_pass(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps({"words": [{"text": "hello", "start": 0.0, "end": 0.2}, {"text": "there", "start": 0.25, "end": 0.5}]}),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {"provider": "fixture", "punctuation": {"1": "Hello, there."}},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert synced[0].text == "Hello, there."


def test_cli_sync_rejects_word_changing_punctuation_with_qc_flag(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    wordstream_path = tmp_path / "episode.wordstream.json"
    providers_path = tmp_path / "providers.yaml"
    out_path = tmp_path / "episode.synced.srt"
    workdir = tmp_path / "work"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello there\n\n", encoding="utf-8")
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    wordstream_path.write_text(
        json.dumps({"words": [{"text": "hello", "start": 0.0, "end": 0.2}, {"text": "there", "start": 0.25, "end": 0.5}]}),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump(
            {
                "asr": {"fixture_path": str(wordstream_path)},
                "llm": {"provider": "fixture", "punctuation": {"1": "Hello, world."}},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(out_path),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
        ],
    )

    assert result.exit_code == 0, result.output
    synced = parse_srt_text(out_path.read_text(encoding="utf-8"))
    assert synced[0].text == "hello there"
    report = json.loads((workdir / "episode" / "qc_report.json").read_text(encoding="utf-8"))
    assert any(flag["kind"] == "invalid_punctuation_change" for flag in report["flags"])
