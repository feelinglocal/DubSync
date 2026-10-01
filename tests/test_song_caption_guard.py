"""A song caption (♪ … ♪) is never rewritten from dialogue audio.

Spoken words inside a caption's divergence become their own insertion; a
caption absent from the voice track keeps source text and timing with one
informational note instead of error-level holds.
"""
from __future__ import annotations

import json

import yaml

from dubsync import pipeline
from dubsync.adjudication_regions import SONG_CAPTION_PREFIX, is_song_caption_cue, protect_song_captions
from dubsync.models import Cue, DivergenceSpan, QCFlag, Word
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


def _cues() -> list[Cue]:
    # ep11 cues 408-412 (case-155) in a compact form. Token indices: 408 -> 0..1,
    # 409 -> 2..4, 410 -> 5..7, 412 -> 8..10.
    return [
        Cue(index=408, start_ms=1_212_000, end_ms=1_213_300, lines=["Vamos lá."]),
        Cue(index=409, start_ms=1_213_500, end_ms=1_218_000, lines=["♪Se pudesse existir♪"]),
        Cue(index=410, start_ms=1_218_500, end_ms=1_223_000, lines=["♪Finalmente faço pazes♪"]),
        Cue(index=412, start_ms=1_228_600, end_ms=1_230_000, lines=["Queria pular nuvens."]),
    ]


def _words(extra: tuple[str, float, float]) -> list[Word]:
    return [Word(text=text, start=start, end=end, confidence=None) for text, start, end in [
        ("Vamos", 1212.50, 1212.80), ("lá.", 1212.82, 1213.28),
        extra,
        ("Queria.", 1228.60, 1228.92), ("pular", 1228.94, 1229.30), ("nuvens.", 1229.32, 1229.90),
    ]]


def _caption_span(asr_text: str, start: float, end: float) -> DivergenceSpan:
    return DivergenceSpan(
        case_id="case-155", cue_ids=[409, 410], srt_text="Se pudesse existir Finalmente faço pazes",
        asr_text=asr_text, start=start, end=end, srt_token_indices=[2, 3, 4, 5, 6, 7], asr_word_indices=[2],
        left_anchor_cue_id=408, right_anchor_cue_id=412, left_anchor_end=1213.28, right_anchor_start=1228.60,
    )


def test_song_caption_cue_detection():
    assert is_song_caption_cue(Cue(index=1, start_ms=0, end_ms=1, lines=["♪Essa décima milésima luz acesa♪"]))
    assert is_song_caption_cue(Cue(index=1, start_ms=0, end_ms=1, lines=["♪ first line", "second line ♪"]))
    assert not is_song_caption_cue(Cue(index=1, start_ms=0, end_ms=1, lines=["Queria pular nas nuvens."]))
    assert not is_song_caption_cue(Cue(index=1, start_ms=0, end_ms=1, lines=["♪♪"]))
    assert not is_song_caption_cue(Cue(index=1, start_ms=0, end_ms=1, lines=["você precisa me contar. ♪O que está na moda♪"]))


def test_spoken_word_inside_a_caption_span_becomes_a_separate_insertion():
    words = _words(("Uhum.", 1220.00, 1220.32))
    span = _caption_span("Uhum.", 1220.00, 1220.32)

    caption, speech = protect_song_captions([span], _cues(), words)

    assert caption.case_id == SONG_CAPTION_PREFIX + "case-155"
    assert (caption.cue_ids, caption.srt_token_indices, caption.asr_word_indices, caption.asr_text) == (
        [409, 410], [2, 3, 4, 5, 6, 7], [], "",
    )
    assert caption.srt_text == span.srt_text
    assert speech.case_id == "case-155"
    assert (speech.cue_ids, speech.srt_text, speech.srt_token_indices) == ([], "", [])
    assert (speech.asr_word_indices, speech.asr_text, speech.start, speech.end) == ([2], "Uhum.", 1220.00, 1220.32)
    assert (speech.left_anchor_cue_id, speech.right_anchor_cue_id) == (408, 412)


def test_asr_duplicate_of_the_neighbouring_word_is_not_spoken_over_a_caption():
    # MAI returned "Queria" twice at 1228.60: once as the next cue's first word,
    # once inside the caption span. That produced the delivered cue "♪Queria♪".
    words = _words(("Queria", 1228.60, 1228.92))
    span = _caption_span("Queria", 1228.60, 1228.92)

    spans = protect_song_captions([span], _cues(), words)

    assert [item.case_id for item in spans] == [SONG_CAPTION_PREFIX + "case-155"]
    assert spans[0].asr_word_indices == [] and spans[0].cue_ids == [409, 410]


def test_mixed_caption_and_dialogue_span_keeps_only_the_dialogue_editable():
    # ep17 case-232: 'minha vida' belongs to caption 562, 'A Lumi' to dialogue cue 563.
    cues = [
        Cue(index=562, start_ms=2_332_980, end_ms=2_335_380, lines=["♪Você é como a minha vida♪"]),
        Cue(index=563, start_ms=2_338_900, end_ms=2_339_900, lines=["A Lumi disse"]),
    ]
    words = [Word(text="Lu", start=2338.96, end=2339.08), Word(text="me", start=2339.20, end=2339.30),
             Word(text="disse", start=2339.40, end=2339.80)]
    span = DivergenceSpan(
        case_id="case-232", cue_ids=[562, 563], srt_text="minha vida A Lumi", asr_text="Lu me",
        start=2338.96, end=2339.30, srt_token_indices=[4, 5, 6, 7], asr_word_indices=[0, 1],
        left_anchor_cue_id=562, right_anchor_cue_id=563, right_anchor_start=2339.40,
    )

    caption, speech = protect_song_captions([span], cues, words)

    assert (caption.case_id, caption.cue_ids, caption.srt_text, caption.asr_word_indices) == (
        SONG_CAPTION_PREFIX + "case-232", [562], "minha vida", [],
    )
    assert (speech.case_id, speech.cue_ids, speech.srt_text, speech.srt_token_indices) == (
        "case-232", [563], "A Lumi", [6, 7],
    )
    assert (speech.asr_text, speech.asr_word_indices) == ("Lu me", [0, 1])


def test_dialogue_only_and_silent_caption_spans_are_untouched():
    dialogue = DivergenceSpan(case_id="case-1", cue_ids=[408], srt_text="lá", asr_text="aqui",
                              srt_token_indices=[1], asr_word_indices=[1], start=1212.82, end=1213.28)
    silent = _caption_span("", 1213.28, 1228.60).model_copy(update={"asr_word_indices": []})

    assert protect_song_captions([dialogue, silent], _cues(), _words(("x", 1220.0, 1220.1))) == [dialogue, silent]


def test_caption_span_is_held_without_an_error_and_without_asking_a_model():
    span = _caption_span("", 1213.28, 1228.60).model_copy(update={"asr_word_indices": []})

    provider_spans, held, flags = pipeline._hold_incomplete_source_insertions(
        [span], {}, missing_audio_cue_ids={409}, song_caption_cue_ids={409, 410},
    )

    assert provider_spans == []
    assert (held[0].verdict, held[0].final_text, held[0].confidence) == ("keep_srt", span.srt_text, 1.0)
    # 409 is absent from the voice track and already carries its per-cue note.
    assert [(flag.kind, flag.severity, flag.cue_ids) for flag in flags] == [("song_lyric_source_kept", "info", [410])]


def _sync(tmp_path, srt: str, words: list[dict[str, object]], responses: dict[str, object]):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(fixture)}, "llm": {"provider": "fixture", "responses": responses},
    }, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"
    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.1),
    )
    return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"]


_SRT = (
    "1\n00:00:01,000 --> 00:00:02,000\nVamos lá agora.\n\n"
    "2\n00:00:05,000 --> 00:00:09,000\n♪Essa décima milésima luz acesa♪\n\n"
    "3\n00:00:12,000 --> 00:00:13,500\nJá está melhor agora?\n"
)


def _episode_words(extra: tuple[str, float, float]) -> list[dict[str, object]]:
    return [{"text": text, "start": start, "end": end, "confidence": None} for text, start, end in [
        ("Vamos", 1.00, 1.30), ("lá", 1.32, 1.50), ("agora.", 1.52, 1.90),
        extra,
        ("Já", 12.00, 12.20), ("está", 12.22, 12.50), ("melhor", 12.52, 12.90), ("agora?", 12.92, 13.40),
    ]]


def _use_audio(final_text: str) -> dict[str, object]:
    return {"case-1": {"case_id": "case-1", "verdict": "use_audio", "final_text": final_text,
                       "confidence": 1.0, "reason": "the audio contains this word"}}


def test_dialogue_word_never_overwrites_a_song_caption(tmp_path):
    # ep17 case-233: '♪Leve-me para fugir♪' was delivered as '♪Uhum.♪'.
    cues, flags = _sync(tmp_path, _SRT, _episode_words(("Uhum.", 7.00, 7.40)), _use_audio("Uhum."))

    by_text = {cue.plain_text: cue for cue in cues}
    assert "♪Essa décima milésima luz acesa♪" in by_text
    caption = by_text["♪Essa décima milésima luz acesa♪"]
    # The spoken reaction is its own cue next to the caption.
    assert "Uhum." in by_text and abs(by_text["Uhum."].start_ms - 7000) <= 34
    # The caption keeps its source start; delivered cues never overlap, so it
    # yields to the spoken line instead of staying on screen over it.
    assert caption.start_ms == 5000 and caption.end_ms == by_text["Uhum."].start_ms
    assert not any("♪Uhum" in cue.plain_text for cue in cues)
    assert any(flag["kind"] == "adlib_inserted" for flag in flags)
    caption_kinds = [flag["kind"] for flag in flags if 2 in flag["cue_ids"]]
    assert caption_kinds.count("song_lyric_source_kept") == 1
    # The caption itself is neither a hold error nor an unmatched-cue finding;
    # only the (real) overlap with the spoken reaction remains reviewable.
    assert not {kind for kind in caption_kinds if "overlap" not in kind} - {"song_lyric_source_kept"}


def test_song_caption_absent_from_the_voice_track_has_one_informational_note(tmp_path):
    words = [word for word in _episode_words(("x", 0.0, 0.1)) if word["text"] != "x"]

    cues, flags = _sync(tmp_path, _SRT, words, {})

    caption = next(cue for cue in cues if "♪" in cue.plain_text)
    assert (caption.plain_text, caption.start_ms, caption.end_ms) == ("♪Essa décima milésima luz acesa♪", 5000, 9000)
    caption_flags = [flag for flag in flags if 2 in flag["cue_ids"]]
    assert [(flag["kind"], flag["severity"]) for flag in caption_flags] == [("song_lyric_source_kept", "info")]
    assert not any(flag["severity"] == "error" for flag in flags)


def test_silence_findings_on_a_noted_song_caption_are_not_repeated():
    note = QCFlag(kind="song_lyric_source_kept", cue_ids=[2], severity="info", message="kept")
    flags = [
        note,
        QCFlag(kind="dropped_line_candidate", cue_ids=[2], message="no speech"),
        QCFlag(kind="cue_without_speech_activity", cue_ids=[2], message="no speech"),
        QCFlag(kind="cue_on_silence", cue_ids=[2], message="silence"),
        QCFlag(kind="cue_without_speech_activity", cue_ids=[3], message="dialogue without speech"),
        QCFlag(kind="dropped_line_candidate", cue_ids=[2, 3], message="mixed finding stays"),
    ]

    kept = pipeline._without_song_caption_silence_duplicates(flags)

    assert [(flag.kind, flag.cue_ids) for flag in kept] == [
        ("song_lyric_source_kept", [2]),
        ("cue_without_speech_activity", [3]),
        ("dropped_line_candidate", [2, 3]),
    ]
