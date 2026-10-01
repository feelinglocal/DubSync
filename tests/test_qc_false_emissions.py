"""QC emitters must not repeat one root cause or fire on facts the audio explains."""
from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
import yaml

from dubsync import pipeline
from dubsync.adjudication import AdjudicationEngine, StaticLLMAdapter
from dubsync.models import AdjudicationDecision, AlignmentDiagnostics, AlignmentResult, Cue, DivergenceSpan, TokenMatch, Word
from dubsync.observability import name_spelling_inconsistency_flags, span_coverage_flags
from dubsync.providers import ProviderError
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


def _span(case_id: str = "case-1", cue_id: int = 1) -> DivergenceSpan:
    return DivergenceSpan(
        case_id=case_id, cue_ids=[cue_id], srt_text="old source words",
        asr_text="different spoken wording", start=0.1, end=0.2, confidence=0.98,
    )


def _sync(tmp_path, srt: str, words: list[dict[str, object]], *, responses=None, no_llm=False):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
    config: dict[str, object] = {"asr": {"fixture_path": str(fixture)}}
    if not no_llm:
        config["llm"] = {"provider": "fixture", "responses": responses or {}}
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"
    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers, no_llm=no_llm,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.1),
    )
    return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"]


# --- one hold, one finding -------------------------------------------------------------------


def test_unavailable_case_audio_is_not_also_a_low_confidence_finding():
    @contextmanager
    def no_snippets(_spans):
        yield {}

    class AudioAdapter:
        def adjudicate(self, spans):
            raise AssertionError("no text-only approval")

        def adjudicate_with_audio(self, spans, snippets):
            raise AssertionError("no clip exists")

    decisions, flags = AdjudicationEngine(
        AudioAdapter(), audio_snippet_batches=no_snippets, require_audio_snippets=True,
    ).adjudicate([_span()])

    assert (decisions[0].verdict, decisions[0].confidence) == ("keep_srt", 0.0)
    assert [flag.kind for flag in flags] == ["adjudication_audio_unavailable"]


def test_provider_failure_is_not_also_a_low_confidence_finding():
    class FailingAdapter:
        def adjudicate(self, spans):
            raise ProviderError("synthetic outage; no external call")

    decisions, flags = AdjudicationEngine(FailingAdapter()).adjudicate([_span()])

    assert decisions[0].verdict == "keep_srt"
    assert [flag.kind for flag in flags] == ["llm_provider_unavailable"]


def test_invalid_response_is_not_also_a_low_confidence_finding():
    decisions, flags = AdjudicationEngine(StaticLLMAdapter({})).adjudicate([_span()])

    assert decisions[0].verdict == "keep_srt"
    assert [flag.kind for flag in flags] == ["invalid_llm_response"]


def test_pipeline_hold_is_not_gated_into_a_second_low_confidence_finding():
    # ep11 cues 441/442: a delete-only span touching a missing-audio cue.
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[441, 442], srt_text="Tao chega", asr_text="",
        start=1301.9, end=1302.2, srt_token_indices=[0, 1],
    )
    _, held, hold_flags = pipeline._hold_incomplete_source_insertions([span], {}, missing_audio_cue_ids={441})

    selected, gate_flags = pipeline._confidence_gate_decisions([span], held, {}, hold_flags)

    assert selected == held
    assert gate_flags == []
    assert [flag.kind for flag in hold_flags] == ["missing_audio_source_cue_held"]
    # Only the cue named by the hold stays protected; its matched neighbour is free.
    assert hold_flags[0].cue_ids == [441]
    assert pipeline._confidence_held_source_cue_ids([*hold_flags, *gate_flags]) == set()


def test_punctuation_only_span_is_not_an_unresolved_divergence_without_llm(tmp_path):
    srt = "1\n00:00:32,800 --> 00:00:34,130\nIch hab's dir doch gesagt.\n"
    words = [{"text": text, "start": start, "end": end, "confidence": None} for text, start, end in [
        ("Ich", 32.72, 32.86), ("hab's", 32.88, 33.10), ("dir", 33.12, 33.26),
        ("doch", 33.28, 33.46), ("gesagt.", 33.48, 33.72),
    ]]

    _, flags = _sync(tmp_path, srt, words, no_llm=True)

    assert "divergence_unresolved" not in [flag["kind"] for flag in flags]


def test_one_ownership_failure_is_reported_once(tmp_path):
    # MAI ep11 cue 114 (case-32): the approved replacement spans three speech
    # groups. The text hold and the word-mapping hold describe the same failure.
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nAntes disso tudo.\n\n"
        "2\n00:00:09,960 --> 00:00:10,880\nLuan Nian!\n\n"
        "3\n00:00:18,000 --> 00:00:19,000\nDepois disso tudo.\n"
    )
    words = [{"text": text, "start": start, "end": end, "confidence": None} for text, start, end in [
        ("Antes", 1.00, 1.20), ("disso", 1.22, 1.50), ("tudo.", 1.52, 1.90),
        ("Hã?", 4.44, 4.72), ("Luanian!", 9.96, 10.72), ("Ei,", 15.20, 15.58),
        ("Depois", 18.00, 18.25), ("disso", 18.27, 18.55), ("tudo.", 18.57, 18.95),
    ]]
    responses = {"case-1": {
        "case_id": "case-1", "verdict": "hybrid", "final_text": "Hã? Luan Nian! Ei,",
        "confidence": 0.95, "reason": "audible reaction, name and interjection",
    }}

    cues, flags = _sync(tmp_path, srt, words, responses=responses)

    assert [cue.plain_text for cue in cues] == ["Antes disso tudo.", "Luan Nian!", "Depois disso tudo."]
    assert (cues[1].start_ms, cues[1].end_ms) == (9960, 10880)
    holds = [flag for flag in flags if flag["kind"] in {
        "adjudication_replacement_ownership_held", "adjudication_word_mapping_held",
    }]
    assert [(flag["kind"], flag["cue_ids"]) for flag in holds] == [("adjudication_replacement_ownership_held", [2])]
    assert not any(str(flag["kind"]).endswith("_source_cue_restored") for flag in flags)


# --- restore flags only when something was restored -----------------------------------------


@pytest.mark.parametrize("reason", ["low_confidence", "timing_evidence"])
def test_timing_hold_with_source_timing_reports_no_restore(reason):
    source = Cue(index=439, start_ms=1000, end_ms=3000, lines=["Vamos subir agora, por favor."])
    # Approved wording or a line re-wrap is retained on purpose; timing is already source.
    held = source.with_lines(["Vamos subir agora,", "por favor, tá?"])

    restored, flags = pipeline._restore_missing_audio_source_cues([held], [source], {439}, reason=reason)

    assert restored == [held]
    assert flags == []


@pytest.mark.parametrize("reason", ["low_confidence", "timing_evidence"])
def test_timing_hold_reports_a_real_timing_restore(reason):
    source = Cue(index=1, start_ms=1000, end_ms=3000, lines=["Source wording."])
    moved = source.with_timing(1200, 2800)

    restored, flags = pipeline._restore_missing_audio_source_cues([moved], [source], {1}, reason=reason)

    assert (restored[0].start_ms, restored[0].end_ms) == (1000, 3000)
    assert [flag.kind for flag in flags] == [f"{reason}_source_cue_restored"]


@pytest.mark.parametrize("reason", ["missing_audio", "protected_region"])
def test_punctuation_only_change_of_a_locked_cue_is_undone_without_an_error(reason):
    source = Cue(index=7, start_ms=1000, end_ms=3000, lines=["♪Essa décima milésima luz acesa♪"])
    punctuated = source.with_lines(["♪Essa décima, milésima luz acesa.♪"])

    restored, flags = pipeline._restore_missing_audio_source_cues([punctuated], [source], {7}, reason=reason)

    assert restored[0].lines == source.lines
    assert flags == []


# --- words the actor really said --------------------------------------------------------------


def _pt_source() -> list[Cue]:
    return parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\nVocê conhece a Lumi?\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nEle conhece o Luan.\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nEu vi o Luan ontem.\n\n"
    )


def test_improvised_word_present_in_the_asr_is_not_an_unsourced_substitution():
    # ep11: 'conhece' -> 'comecei' was spoken; the detector never looked at the audio.
    output = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nEu comecei com a Lumi.\n\n")
    words = [Word(text=text, start=start, end=start + 0.2) for text, start in [
        ("Eu", 1.0), ("comecei", 1.2), ("com", 1.4), ("a", 1.6), ("Lumi.", 1.8),
    ]]

    assert [flag.kind for flag in name_spelling_inconsistency_flags(_pt_source(), output)] == ["unsourced_word_substitution"]
    assert name_spelling_inconsistency_flags(_pt_source(), output, asr_words=words) == []


def test_word_absent_from_the_asr_is_still_an_unsourced_substitution():
    output = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nEu comecei com a Lumi.\n\n")
    words = [Word(text=text, start=start, end=start + 0.2) for text, start in [
        ("Eu", 1.0), ("conheço", 1.2), ("a", 1.6), ("Lumi.", 1.8),
        ("comecei", 55.0),  # the same word far away is not evidence for this cue
    ]]

    flags = name_spelling_inconsistency_flags(_pt_source(), output, asr_words=words)

    assert [(flag.kind, flag.cue_ids) for flag in flags] == [("unsourced_word_substitution", [1])]


def test_asr_spelling_of_a_source_name_is_still_reported_as_name_drift():
    # ep17: 'Luan' -> 'Luanyan' comes from the ASR itself; the human keeps the source spelling.
    output = parse_srt_text("1\n00:00:05,000 --> 00:00:06,000\nEu vi o Luanyan ontem.\n\n")
    words = [Word(text=text, start=start, end=start + 0.2) for text, start in [
        ("Eu", 5.0), ("vi", 5.2), ("o", 5.4), ("Luanyan", 5.5), ("ontem.", 5.8),
    ]]

    flags = name_spelling_inconsistency_flags(_pt_source(), output, asr_words=words)

    assert [flag.kind for flag in flags] == ["name_spelling_inconsistency"]


def test_function_word_capitalized_once_is_not_a_name():
    # ep11: 'ela' (29x, once written 'Ela' after a comma) -> 'Elas' was reported as name drift.
    source = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\nEu vi ela ontem.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nMas ela não veio.\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nOlha, Ela chegou e ela ficou.\n\n"
    )
    output = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nEu vi elas ontem.\n\n")
    heard = [Word(text="elas", start=1.4, end=1.6)]

    assert [flag.kind for flag in name_spelling_inconsistency_flags(source, output)] == ["unsourced_word_substitution"]
    assert name_spelling_inconsistency_flags(source, output, asr_words=heard) == []


# --- coverage of an approved deletion ---------------------------------------------------------


def _coverage(source: list[Cue], rebuilt: list[Cue], final_text: str):
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[cue.index for cue in source],
        srt_text=" ".join(cue.plain_text for cue in source), asr_text=final_text, start=10.0, end=11.0,
    )
    decision = AdjudicationDecision(
        case_id="case-1", verdict="use_audio", final_text=final_text, confidence=0.95, reason="fixture",
    )
    return span_coverage_flags(source, rebuilt, [span], [decision])


def test_shortened_dialogue_is_not_low_span_coverage():
    # ep11 cue 490: six source words, the actor only says one.
    source = [Cue(index=490, start_ms=10_000, end_ms=12_000, lines=["Mas hoje não estou muito bem."])]
    rebuilt = [Cue(index=490, start_ms=10_100, end_ms=10_600, lines=["É."])]

    assert _coverage(source, rebuilt, "É.") == []


def test_completely_deleted_span_is_not_low_span_coverage():
    source = [Cue(index=3, start_ms=10_000, end_ms=12_000, lines=["Not spoken at all."])]

    assert _coverage(source, [], "") == []


def test_same_words_squeezed_into_a_fraction_of_the_time_is_low_span_coverage():
    source = [Cue(index=5, start_ms=10_000, end_ms=14_000, lines=["alpha beta gamma delta"])]
    rebuilt = [Cue(index=5, start_ms=10_000, end_ms=10_800, lines=["alpha beta gamma omega"])]

    flags = _coverage(source, rebuilt, "alpha beta gamma omega")

    assert [flag.kind for flag in flags] == ["span_coverage_low"]


# --- anchor coverage on lyric-bearing episodes ------------------------------------------------


def _lyric_episode() -> tuple[list[Cue], AlignmentResult]:
    cues = [
        Cue(index=1, start_ms=0, end_ms=4000, lines=["♪Opening song line one two three♪"]),
        Cue(index=2, start_ms=4000, end_ms=8000, lines=["♪Opening song line four five six♪"]),
        Cue(index=3, start_ms=9000, end_ms=10000, lines=["Hello there my friend."]),
        Cue(index=4, start_ms=10000, end_ms=11000, lines=["Good to see you."]),
    ]
    matches = [TokenMatch(cue_id=3, srt_token_index=12 + i, asr_word_index=i, score=1.0) for i in range(4)]
    matches += [TokenMatch(cue_id=4, srt_token_index=16 + i, asr_word_index=4 + i, score=1.0) for i in range(4)]
    # 8 of 20 source tokens matched: the 12 unmatched tokens are song captions.
    alignment = AlignmentResult(
        token_matches=matches, anchor_coverage=0.4, unmatched_cue_ids=[1, 2],
        diagnostics=AlignmentDiagnostics(missing_audio_cue_ids=[1, 2]),
    )
    return cues, alignment


def test_song_captions_absent_from_the_voice_track_do_not_lower_anchor_coverage():
    cues, alignment = _lyric_episode()

    assert pipeline._alignment_health_flags(alignment, source_cue_count=4, source_cues=cues) == []


def test_unmatched_dialogue_still_lowers_anchor_coverage():
    cues, alignment = _lyric_episode()
    cues = [cue.with_lines([cue.plain_text.replace("♪", "")]) for cue in cues]

    flags = pipeline._alignment_health_flags(alignment, source_cue_count=4, source_cues=cues)

    assert [(flag.kind, flag.severity) for flag in flags] == [("alignment_anchor_coverage_low", "error")]
