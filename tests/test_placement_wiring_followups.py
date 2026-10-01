"""Wiring follow-ups of the accuracy upgrade: language, ASR evidence, lexical support, speakers."""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.models import AlignmentResult, Cue, Word
from dubsync.output_order import finalize_cues_for_output
from dubsync.overlap import apply_overlap_policy
from dubsync.recue import rebuild_cues, timing_evidence_issue
from dubsync.style_profile import StyleProfile
from dubsync.verify import lint_cues


def _sync(tmp_path, srt: str, words: list[dict[str, object]], *, asr: dict[str, object] | None = None, **kwargs):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture), **(asr or {})}}, allow_unicode=True), encoding="utf-8")
    result = pipeline.sync_episode(
        source, audio, tmp_path / "out.srt", tmp_path / "work", providers_path=providers, no_llm=True,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.5), **kwargs,
    )
    return result


_UM_SRT = "1\n00:00:01,000 --> 00:00:02,500\nEr kommt um acht.\n"
_UM_WORDS = [
    {"text": "Er", "start": 1.0, "end": 1.2, "confidence": None}, {"text": "kommt", "start": 1.25, "end": 1.6, "confidence": None},
    {"text": "1", "start": 1.65, "end": 1.8, "confidence": None}, {"text": "acht.", "start": 1.85, "end": 2.3, "confidence": None},
]


# --- (a) episode language reaches the aligner -------------------------------------------------


@pytest.mark.parametrize("job_language, asr_config, divergences", [
    ("de", None, 1),                      # German "um" is not the number one
    ("deu", None, 1),                     # ISO-639-3 from the web form
    (None, {"language_code": "de"}, 1),   # configured for the ASR provider only
    ("pt", None, 0),                      # Portuguese "um" is
    (None, None, 0),                      # unknown language: every alias applies
])
def test_alignment_resolves_number_words_in_the_episode_language(tmp_path, job_language, asr_config, divergences):
    result = _sync(tmp_path, _UM_SRT, _UM_WORDS, asr=asr_config, language=job_language)

    alignment = json.loads((result.episode_workdir / "align.json").read_text(encoding="utf-8"))
    assert len(alignment["divergence_spans"]) == divergences


# --- (b) provider evidence is saved with the sync transcript ----------------------------------


def test_sync_asr_artifact_keeps_the_adapter_provider_evidence(tmp_path, monkeypatch):
    evidence = {"provider": "openrouter", "chunks": [{"language": "de"}], "speaker_links": []}

    class EvidenceAdapter:
        last_usage: dict[str, object] = {}
        last_repair_flags: list = []
        last_evidence = evidence

        def transcribe(self, audio_path):
            return [Word(**item) for item in _UM_WORDS]

    monkeypatch.setattr(pipeline, "adapter_from_config", lambda *args, **kwargs: EvidenceAdapter())

    result = _sync(tmp_path, _UM_SRT, _UM_WORDS)

    metadata = json.loads((result.episode_workdir / "asr.json").read_text(encoding="utf-8"))["metadata"]
    assert metadata["provider_evidence"] == evidence


def test_sync_asr_artifact_has_no_evidence_key_for_an_adapter_without_evidence(tmp_path):
    result = _sync(tmp_path, _UM_SRT, _UM_WORDS)

    metadata = json.loads((result.episode_workdir / "asr.json").read_text(encoding="utf-8"))["metadata"]
    assert "provider_evidence" not in metadata


# --- (c) compounds and numbers are lexical timing evidence ------------------------------------


def _timed(texts: list[str], start: float = 2.0) -> list[Word]:
    return [Word(text=text, start=start + index * 0.4, end=start + index * 0.4 + 0.3, confidence=None)
            for index, text in enumerate(texts)]


@pytest.mark.parametrize("lines, spoken", [
    # examples-dialogue MAI: the hyphenated line break is spoken as one compound word.
    (["Drachen-", "Evolutionssystem besitze."], ["Drachen-Evolutionssystem", "besitze."]),
    (["Feliz Ano-Novo, pessoal."], ["Feliz", "ano", "novo,", "pessoal."]),
    (["Tenho vinte e seis anos."], ["Tenho", "26", "anos."]),
    (["O voo 190 saiu."], ["O", "voo", "um", "nove", "zero", "saiu."]),
    (["Dez, nove, oito."], ["10,", "9,", "8."]),
])
def test_compound_and_number_spellings_count_as_lexical_timing_evidence(lines, spoken):
    cue = Cue(index=1, start_ms=0, end_ms=1500, lines=lines)
    words = _timed(spoken)

    assert timing_evidence_issue(cue, words) is None
    rebuilt, flags = rebuild_cues(
        [cue], words, AlignmentResult(cue_word_indices={1: list(range(len(words)))}), StyleProfile(min_cue_dur=0.1, tail_ms=0),
    )
    assert [flag.kind for flag in flags] == [] and rebuilt[0].start_ms == 2000


def test_unrelated_words_are_still_sparse_lexical_evidence():
    cue = Cue(index=1, start_ms=0, end_ms=1500, lines=["Drachen-", "Evolutionssystem besitze."])

    assert "Sparse lexical" in (timing_evidence_issue(cue, _timed(["Drachen", "kommen."])) or "")
    assert "Sparse lexical" in (timing_evidence_issue(
        Cue(index=2, start_ms=0, end_ms=1500, lines=["Tenho vinte e seis anos."]), _timed(["27"]),
    ) or "")


# --- (d) only provably different speakers count as different ----------------------------------


def _pair(left_speaker: str | None, right_speaker: str | None, *, same_start: bool = False) -> list[Cue]:
    return [
        Cue(index=1, start_ms=1000, end_ms=2500, lines=["Eu vou agora."], speaker_id=left_speaker),
        Cue(index=2, start_ms=1000 if same_start else 2000, end_ms=3200,
            lines=["Eu vou agora." if same_start else "Fica aqui."], speaker_id=right_speaker),
    ]


def test_labels_of_unrelated_chunk_scopes_are_not_two_speakers_for_a_dash_merge():
    merged, flags = apply_overlap_policy(_pair("chunk_1:0", "chunk_2:0"), "dash")
    assert len(merged) == 2 and "overlap_dash_merge" not in [flag.kind for flag in flags]

    merged, flags = apply_overlap_policy(_pair("chunk_1:0", "chunk_1:1"), "dash")
    assert len(merged) == 1 and flags[0].kind == "overlap_dash_merge"


def test_overlap_of_unrelated_chunk_scopes_is_still_a_style_issue():
    kinds = lambda cues: [issue.kind for issue in lint_cues(cues, StyleProfile(fps=1000, min_cue_dur=0.1))]  # noqa: E731

    assert "overlap" in kinds(_pair("chunk_1:0", "chunk_2:0"))
    assert "overlap" in kinds(_pair(None, "speaker_1"))
    assert "overlap" not in kinds(_pair("chunk_1:0", "chunk_1:1"))
    assert "overlap" not in kinds(_pair("speaker_1", "speaker_2"))


def test_duplicate_cue_of_unrelated_chunk_scopes_is_merged():
    profile = StyleProfile(fps=1000, min_cue_dur=0.1)

    merged, flags = finalize_cues_for_output(_pair("chunk_1:0", "chunk_2:1", same_start=True), profile, preserve_timing=True)
    assert len(merged) == 1 and [flag.kind for flag in flags] == ["duplicate_cue_merged"]

    kept, _ = finalize_cues_for_output(_pair("chunk_1:0", "chunk_1:1", same_start=True), profile, preserve_timing=True)
    assert len(kept) == 2
