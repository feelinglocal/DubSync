from __future__ import annotations

import pytest

from dubsync.aligner import align_cues_to_words
from dubsync.adjudication import AdjudicationEngine
from dubsync.adjudication_regions import split_protected_source_repetitions, validated_protected_source_regions
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, AudioSnippet, Cue, Word
from dubsync.pipeline import (
    _adlib_cue_ids_by_case, _alignment_with_decision_words,
    _apply_incomplete_source_holds_to_decisions, _confidence_gate_decisions,
    _hold_incomplete_source_insertions, _timing_evidence_held_cue_ids,
    _validate_alignment_screen_text_provenance, _validate_rebuild_policy,
)
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature, tokenize_cues


def actual_third_greeting():
    # Original episode11 cues730–737 and full MAI words3011–3023. These
    # independent acoustic occurrences are not inferred from human captions.
    cues = [
        Cue(index=730, start_ms=2038070, end_ms=2038680, lines=["Saúde."]),
        Cue(index=731, start_ms=2038960, end_ms=2040070, lines=["Feliz ano novo."]),
        Cue(index=732, start_ms=2040070, end_ms=2040990, lines=["Feliz ano novo!"]),
        Cue(index=733, start_ms=2042620, end_ms=2047300, lines=["♪Essas inquietações e feridas estão aos poucos cicatrizando♪"]),
        Cue(index=734, start_ms=2047740, end_ms=2053380, lines=["♪Mesmo que a tempestade cubra o porto da vinda♪"]),
        Cue(index=735, start_ms=2055620, end_ms=2061700, lines=["♪Haverá luz nas estrelas piscando nos olhos♪"]),
        Cue(index=736, start_ms=2061700, end_ms=2062500, lines=["[IMAX bar]"]),
        Cue(index=737, start_ms=2067320, end_ms=2068470, lines=["Verificação bem-sucedida."]),
    ]
    values = [
        ("Saúde!", 2038.2, 2038.6390000000001),
        ("Feliz", 2038.88, 2039.06), ("Ano", 2039.08, 2039.219),
        ("Novo!", 2039.24, 2039.339), ("Feliz", 2039.3600000000001, 2039.459),
        ("Ano", 2039.48, 2039.72), ("Novo!", 2039.76, 2040.0),
        ("Feliz", 2040.12, 2040.3600000000001), ("Ano", 2040.4, 2040.58),
        ("Novo.", 2040.68, 2041.039), ("Verificação", 2067.36, 2067.959),
        ("bem", 2068.08, 2068.18), ("sucedida.", 2068.24, 2068.779),
    ]
    return cues, [Word(text=text, start=start, end=end, confidence=None, speaker_id=None)
                  for text, start, end in values]


def test_extra_complete_greeting_is_separate_from_later_protected_lyrics():
    cues, words = actual_third_greeting()
    original = [cue.model_dump() for cue in cues]

    alignment = align_cues_to_words(cues, words)

    assert [span.case_id for span in alignment.divergence_spans] == [
        "protected-source-case-1", "speech-repeat-case-1", "case-2",
    ]
    source, speech = alignment.divergence_spans[:2]
    assert source.cue_ids == [733, 734, 735]
    assert source.srt_token_indices == list(range(7, 31))
    assert not source.asr_text and not source.asr_word_indices
    assert (source.start, source.end) == (2042.62, 2061.7)
    assert speech.cue_ids == [] and speech.srt_token_indices == [] and speech.srt_text == ""
    assert speech.asr_text == "Feliz Ano Novo."
    assert speech.asr_word_indices == [7, 8, 9]
    assert (speech.start, speech.end) == (words[7].start, words[9].end)
    assert (speech.left_anchor_cue_id, speech.right_anchor_cue_id) == (732, 737)
    assert (speech.left_anchor_end, speech.right_anchor_start) == (words[6].end, words[10].start)
    assert speech.insertion_token_offset is None
    assert alignment.cue_word_indices == {730: [0], 731: [1, 2, 3], 732: [4, 5, 6], 737: [10]}
    assert alignment.diagnostics.excluded_screen_text_cue_ids == [736]
    assert [cue.model_dump() for cue in cues] == original


def classification(alignment, cues, words):
    return validated_protected_source_regions(
        alignment.divergence_spans, alignment.token_matches, cues, words,
        protected_cue_ids=set(alignment.diagnostics.missing_audio_cue_ids),
    )


def approval(span, **updates):
    return AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
        confidence=0.99, reason="Explicit fresh diagnostic audio approval",
    ).model_copy(update=updates)


def test_recovery_is_not_specific_to_the_reported_greeting():
    cues, words = actual_third_greeting()
    for index in (1, 2):
        cues[index] = cues[index].with_lines(["Muito obrigado amigo!"])
    for first in (1, 4, 7):
        for offset, text in enumerate(("Muito", "obrigado", "amigo!")):
            words[first + offset] = words[first + offset].model_copy(update={"text": text})
    alignment = align_cues_to_words(cues, words)
    assert classification(alignment, cues, words) == {"protected-source-case-1": {733, 734, 735}}
    assert alignment.divergence_spans[1].asr_text == "Muito obrigado amigo!"


def raw_alignment(cues, words, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr("dubsync.aligner.split_protected_source_repetitions", lambda spans, *_args, **_kwargs: spans)
        return align_cues_to_words(cues, words)


@pytest.mark.parametrize("problem", [
    "different_phrase", "partial_phrase", "one_word", "lyrics_in_asr", "noise_word",
    "mixed_music_speech", "unmarked_source", "mixed_annotation", "partial_source", "annotation_footprint",
    "overlapping_source", "later_source_overlap", "invalid_source", "prior_music", "prior_annotation",
    "missing_prior_match", "fuzzy_prior_match", "forged_prior_word", "forged_prior_cue",
    "duplicate_prior_match", "shared_new_word", "shared_source", "shared_prior_word",
    "prior_protected", "right_protected", "source_protected", "gap", "internal_gap", "word_overlap",
    "invalid_bounds", "invalid_confidence", "low_confidence", "mixed_speakers", "anchor_speaker_conflict",
    "wrong_anchor_time", "wrong_right_anchor", "missing_right_anchor", "nonconsecutive_words",
    "wrong_span_text", "wrong_span_time", "repeated_loop",
])
def test_ambiguous_or_incomplete_evidence_keeps_original_replacement(problem, monkeypatch):
    cues, words = actual_third_greeting()
    raw = raw_alignment(cues, words, monkeypatch)
    spans, matches = list(raw.divergence_spans), list(raw.token_matches)
    span = spans[0]
    protected = set()
    if problem == "different_phrase":
        words[7] = words[7].model_copy(update={"text": "Bom"})
        span = span.model_copy(update={"asr_text": "Bom Ano Novo."})
    elif problem in {"partial_phrase", "one_word"}:
        indices = [7, 8] if problem == "partial_phrase" else [9]
        span = span.model_copy(update={"asr_word_indices": indices,
            "asr_text": " ".join(words[index].text for index in indices),
            "start": words[indices[0]].start, "end": words[indices[-1]].end})
    elif problem in {"lyrics_in_asr", "noise_word"}:
        words[8] = words[8].model_copy(update={"text": "♪Ano♪" if problem == "lyrics_in_asr" else "..."})
        span = span.model_copy(update={"asr_text": " ".join(item.text for item in words[7:10])})
    elif problem in {"mixed_music_speech", "unmarked_source", "mixed_annotation"}:
        lines = (["♪Essas inquietações♪ Ordinary speech ♪e feridas♪"] if problem == "mixed_music_speech"
                 else [cues[3].text.strip("♪")] if problem == "unmarked_source"
                 else ["[SIGN]", cues[3].text])
        cues[3] = cues[3].with_lines(lines)
    elif problem == "partial_source":
        span = span.model_copy(update={"srt_token_indices": span.srt_token_indices[:-1]})
    elif problem == "annotation_footprint":
        span = span.model_copy(update={"cue_ids": [*span.cue_ids, 736]})
    elif problem in {"overlapping_source", "later_source_overlap", "invalid_source"}:
        index = 4 if problem == "later_source_overlap" else 3
        cues[index] = cues[index].model_copy(update={"start_ms": 2040000 if problem != "invalid_source" else cues[index].end_ms})
    elif problem in {"prior_music", "prior_annotation"}:
        cues[2] = cues[2].with_lines(["♪Feliz ano novo!♪"] if problem == "prior_music" else ["[PERSON]", "Feliz ano novo!"])
    elif problem == "missing_prior_match":
        matches.pop(5)
    elif problem in {"fuzzy_prior_match", "forged_prior_word", "forged_prior_cue"}:
        update = {"score": 0.99} if problem == "fuzzy_prior_match" else {"asr_word_index": 2} if problem == "forged_prior_word" else {"cue_id": 731}
        matches[5] = matches[5].model_copy(update=update)
    elif problem == "duplicate_prior_match":
        matches.append(matches[5])
    elif problem in {"shared_new_word", "shared_source", "shared_prior_word"}:
        other = spans[-1].model_copy(update={"case_id": "competing"})
        update = {"asr_word_indices": [8]} if problem == "shared_new_word" else {"srt_token_indices": [8]} if problem == "shared_source" else {"asr_word_indices": [5]}
        spans.append(other.model_copy(update=update))
    elif problem in {"prior_protected", "right_protected", "source_protected"}:
        protected = {732 if problem == "prior_protected" else 737 if problem == "right_protected" else 734}
    elif problem == "gap":
        words[6] = words[6].model_copy(update={"end": 2039.8})
        span = span.model_copy(update={"left_anchor_end": words[6].end})
    elif problem in {"internal_gap", "word_overlap"}:
        words[8] = words[8].model_copy(update={"start": 2040.59 if problem == "internal_gap" else 2040.3, "end": 2040.64})
    elif problem in {"invalid_bounds", "invalid_confidence", "low_confidence"}:
        update = {"end": float("nan")} if problem == "invalid_bounds" else {"confidence": float("nan")} if problem == "invalid_confidence" else {"confidence": 0.79}
        words[8] = words[8].model_copy(update=update)
    elif problem == "mixed_speakers":
        words[7] = words[7].model_copy(update={"speaker_id": "A"})
        words[8] = words[8].model_copy(update={"speaker_id": "B"})
    elif problem == "anchor_speaker_conflict":
        span = span.model_copy(update={"left_anchor_speaker_id": "A"})
    elif problem == "wrong_anchor_time":
        span = span.model_copy(update={"left_anchor_end": 2040.1})
    elif problem == "wrong_right_anchor":
        span = span.model_copy(update={"right_anchor_cue_id": 736})
    elif problem == "missing_right_anchor":
        matches.pop()
    elif problem == "nonconsecutive_words":
        span = span.model_copy(update={"asr_word_indices": [7, 9]})
    elif problem == "wrong_span_text":
        span = span.model_copy(update={"asr_text": "Feliz Ano Novo. Extra"})
    elif problem == "wrong_span_time":
        span = span.model_copy(update={"end": 2041.1})
    else:
        cues[2] = cues[2].with_lines(["Ha ha ha!"])
        for index in range(4, 10):
            words[index] = words[index].model_copy(update={"text": "Ha"})
        span = span.model_copy(update={"asr_text": "Ha Ha Ha"})
    spans[0] = span
    before = [item.model_dump() for item in spans]

    result = split_protected_source_repetitions(spans, matches, cues, tokenize_cues(cues), words, protected_cue_ids=protected)

    assert [item.model_dump() for item in result] == before
    assert [item.model_dump() for item in spans] == before


@pytest.mark.parametrize("problem", ["source_audio", "source_text", "source_bounds", "orphan_source", "orphan_speech", "speech_source", "speech_bounds", "old_case_collision"])
def test_reserved_child_ids_cannot_bypass_evidence_validation(problem):
    cues, words = actual_third_greeting()
    alignment = align_cues_to_words(cues, words)
    spans = list(alignment.divergence_spans)
    if problem == "source_audio":
        spans[0] = spans[0].model_copy(update={"asr_text": "invented audio", "asr_word_indices": [7]})
    elif problem == "source_text":
        spans[0] = spans[0].model_copy(update={"srt_text": "arbitrary protected words"})
    elif problem == "source_bounds":
        spans[0] = spans[0].model_copy(update={"start": 2040.0})
    elif problem.startswith("orphan"):
        spans.pop(1 if problem == "orphan_source" else 0)
    elif problem == "speech_source":
        spans[1] = spans[1].model_copy(update={"cue_ids": [733]})
    elif problem == "speech_bounds":
        spans[1] = spans[1].model_copy(update={"end": 2050.0})
    else:
        spans.append(spans[0].model_copy(update={"case_id": "case-1"}))
    with pytest.raises(ValueError, match="resume from align"):
        validated_protected_source_regions(spans, alignment.token_matches, cues, words, protected_cue_ids=set())


def test_derived_region_split_is_idempotent():
    cues, words = actual_third_greeting()
    alignment = align_cues_to_words(cues, words)
    assert split_protected_source_repetitions(alignment.divergence_spans, alignment.token_matches,
        cues, tokenize_cues(cues), words, protected_cue_ids=set()) == alignment.divergence_spans
    assert classification(alignment, cues, words) == {"protected-source-case-1": {733, 734, 735}}


@pytest.mark.parametrize("proposed_text", ["", "Model rewrote the protected song lyrics"])
def test_validated_source_branch_is_preserved_before_dispatch_and_overrides_cached_model_edits(proposed_text):
    cues, words = actual_third_greeting()
    alignment = align_cues_to_words(cues, words)
    source, speech = alignment.divergence_spans[:2]
    protected = classification(alignment, cues, words)
    provider_spans, held, flags = _hold_incomplete_source_insertions(
        alignment.divergence_spans, {}, protected_source_regions=protected,
    )
    assert source not in provider_spans and speech in provider_spans
    assert [(item.case_id, item.verdict, item.final_text) for item in held] == [(source.case_id, "keep_srt", source.srt_text)]
    assert any(flag.kind == "protected_source_region_held" and flag.cue_ids == [733, 734, 735] for flag in flags)
    assert not any("missing_audio" in flag.kind for flag in flags)
    selected, _ = _apply_incomplete_source_holds_to_decisions(
        alignment.divergence_spans, {}, [approval(source, final_text=proposed_text)], [],
        protected_source_regions=protected,
    )
    assert selected == held


@pytest.mark.parametrize("approval_kind", ["fresh", "absent", "old_parent", "keep_srt", "low_confidence"])
def test_every_spoken_occurrence_requires_its_new_approval_and_lyrics_survive_remove_policy(approval_kind):
    cues, words = actual_third_greeting()
    alignment = align_cues_to_words(cues, words)
    source, speech = alignment.divergence_spans[:2]
    decisions = [] if approval_kind == "absent" else [approval(speech)]
    if approval_kind == "old_parent":
        decisions = [approval(speech, case_id="case-1")]
    elif approval_kind == "keep_srt":
        decisions = [approval(speech, verdict="keep_srt", final_text="")]
    elif approval_kind == "low_confidence":
        decisions = [approval(speech, confidence=0.3)]
    decisions, flags = _apply_incomplete_source_holds_to_decisions(
        alignment.divergence_spans, {}, decisions, [],
        protected_source_regions=classification(alignment, cues, words),
    )
    decisions, confidence_flags = _confidence_gate_decisions(alignment.divergence_spans, decisions, {}, flags)
    protected = _timing_evidence_held_cue_ids([*flags, *confidence_flags])
    adlib_ids, _ = _adlib_cue_ids_by_case(cues, alignment.divergence_spans, decisions, alignment.unmatched_cue_ids)
    profile = StyleProfile(drop_policy="remove", min_cue_dur=0.1, lead_in_ms=0, tail_ms=0)
    changed, _ = apply_adjudication_decisions(cues, alignment.divergence_spans, decisions, profile,
        adlib_ids, words=words, token_matches=alignment.token_matches)
    mapped = _alignment_with_decision_words(alignment, decisions, alignment.divergence_spans,
        adlib_ids, source_cues=cues, words=words)
    rebuilt, _ = rebuild_cues(changed, words, mapped, profile, protected_cue_ids=protected)

    assert [cue for cue in rebuilt if 733 <= cue.index <= 736] == cues[3:7]
    if approval_kind == "fresh":
        new_id = adlib_ids[speech.case_id]
        extra = next(cue for cue in rebuilt if cue.index == new_id)
        assert extra.text == "Feliz Ano Novo."
        assert (extra.start_ms, extra.end_ms) == (profile.snap_floor(words[7].start * 1000), profile.snap_ceil(words[9].end * 1000))
        assert [index for cue_id in (731, 732, new_id) for index in mapped.cue_word_indices[cue_id]] == list(range(1, 10))
        assert all(sum(index in indices for indices in mapped.cue_word_indices.values()) == 1 for index in range(1, 10))
    else:
        assert not adlib_ids
        assert not any(index in indices for indices in mapped.cue_word_indices.values() for index in (7, 8, 9))


class RecordingAdapter:
    def __init__(self):
        self.calls = []

    def adjudicate(self, spans):
        self.calls.append(("text", [span.case_id for span in spans]))
        return [approval(span).model_dump() for span in spans]

    def adjudicate_with_audio(self, spans, snippets):
        self.calls.append(("audio", [span.case_id for span in spans]))
        return [approval(span).model_dump() for span in spans]


def test_only_the_derived_speech_case_requires_audio_and_missing_audio_cannot_autoapprove():
    cues, words = actual_third_greeting()
    alignment = align_cues_to_words(cues, words)
    speech, ordinary = alignment.divergence_spans[1:]
    adapter = RecordingAdapter()
    required_ids = {speech.case_id}
    engine = AdjudicationEngine(adapter, confidence_gate=0, scene_gap_seconds=100,
                               required_audio_case_ids=required_ids)
    required_ids.clear()  # Constructor must own an immutable copy.
    decisions, flags = engine.adjudicate([speech, ordinary])
    assert [(item.case_id, item.verdict) for item in decisions] == [(speech.case_id, "keep_srt"), (ordinary.case_id, "use_audio")]
    assert adapter.calls == [("text", [ordinary.case_id])]
    assert any(flag.kind == "adjudication_audio_unavailable" and flag.new_text == speech.asr_text for flag in flags)


def test_old_alignment_and_rebuild_versions_cannot_bypass_the_new_region(tmp_path):
    import json
    cues, words = actual_third_greeting()
    alignment = align_cues_to_words(cues, words)
    alignment.diagnostics.missing_audio_guard_version = 5
    with pytest.raises(RuntimeError):
        _validate_alignment_screen_text_provenance(alignment, cues)
    checkpoint = tmp_path / "rebuild.json"
    checkpoint.write_text(json.dumps({"policy_version": 7, "cues": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="resume from rebuild"):
        _validate_rebuild_policy(checkpoint)


@pytest.mark.parametrize("extra_clip_available", [True, False])
def test_pipeline_preserves_source_branches_on_normal_rebuild_and_verify_runs(tmp_path, monkeypatch, extra_clip_available):
    import json
    import socket
    import wave
    import yaml
    from dubsync import pipeline
    from dubsync.srt_io import write_srt

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Protected-region pipeline test cannot use the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "speaker_mapping_adapter_from_config", lambda *_args, **_kwargs: None)
    cues, words = actual_third_greeting()
    # Shift the exact relative fixture to a short local test recording. This
    # avoids large test media while retaining every measured gap and word ID.
    cues = [cue.with_timing(cue.start_ms - 2037000, cue.end_ms - 2037000) for cue in cues]
    words = [word.model_copy(update={"start": word.start - 2037, "end": word.end - 2037}) for word in words]
    source = tmp_path / "episode.srt"
    source.write_text(write_srt(cues), encoding="utf-8")
    audio = tmp_path / "episode.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 16000 * 35)
    wordstream = tmp_path / "words.json"
    wordstream.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(wordstream)},
        "llm": {"provider": "fixture", "adjudication": {
            "audio_context": {"enabled": False}, "audio_snippet_double_check": {"enabled": True},
        }}, "punctuation": {"enabled": False},
    }), encoding="utf-8")
    adapter = RecordingAdapter()
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *_args, **_kwargs: adapter)

    def snippets(_audio, spans, _directory, **_kwargs):
        _directory.mkdir(parents=True, exist_ok=True)
        result = []
        for span in spans:
            if not extra_clip_available and span.case_id.startswith("speech-repeat-"):
                continue
            path = _directory / f"{span.case_id}.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\x00\x00" * round(16000 * (span.end - span.start)))
            result.append(AudioSnippet(case_id=span.case_id, path=str(path), start=span.start, end=span.end))
        return result

    monkeypatch.setattr(pipeline, "extract_audio_snippets", snippets)
    options = dict(srt_path=source, audio_path=audio, output_path=tmp_path / "output.srt",
        workdir=tmp_path / "work", providers_path=config,
        style_profile=StyleProfile(drop_policy="remove", min_cue_dur=0.1, lead_in_ms=0, tail_ms=0))
    result = pipeline.sync_episode(**options)
    episode_dir = result.episode_workdir
    original_source = [Cue.model_validate(item) for item in json.loads((episode_dir / "ingest.json").read_text(encoding="utf-8"))["cues"]]
    song_cues = [cue for cue in original_source if "♪" in cue.text]

    def assert_output():
        rebuilt = [Cue.model_validate(item) for item in json.loads((episode_dir / "rebuild.json").read_text(encoding="utf-8"))["cues"]]
        assert [cue for cue in rebuilt if "♪" in cue.text] == song_cues
        spoken_greetings = sum(len(alphanumeric_signature(cue.text)) // 3 for cue in rebuilt
                              if alphanumeric_signature(cue.text) == ["feliz", "ano", "novo"])
        assert spoken_greetings == (3 if extra_clip_available else 2)
        assert any(flag["kind"] == "protected_source_region_held" for flag in json.loads((episode_dir / "qc_report.json").read_text(encoding="utf-8"))["flags"])

    assert_output()
    assert not any(case_id.startswith("protected-source-") for _, ids in adapter.calls for case_id in ids)
    before_output = result.output_srt.read_bytes()
    pipeline.sync_episode(**options, resume="verify")
    assert result.output_srt.read_bytes() == before_output

    # A stale or altered source-child reply is corrected on rebuild and can
    # never delete its protected source, even under the remove policy.
    path = episode_dir / "adjudicate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    held = next(item for item in payload["decisions"] if item["case_id"].startswith("protected-source-"))
    held.update(verdict="use_audio", final_text="", confidence=1)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume from rebuild"):
        pipeline.sync_episode(**options, resume="verify")
    assert result.output_srt.read_bytes() == before_output
    pipeline.sync_episode(**options, resume="rebuild")
    assert_output()
    assert result.output_srt.read_bytes() == before_output
