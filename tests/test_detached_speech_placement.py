"""Words spoken seconds away from a cue are never shown inside that cue."""
from __future__ import annotations

import json
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.aligner import align_cues_to_words
from dubsync.audio_snippets import _snippet_window
from dubsync.detached_speech import DETACHED_SPEECH_PREFIX, separate_detached_speech, separate_unheard_cue_edges
from dubsync.models import (
    AdjudicationDecision, AlignmentResult, AudioSnippet, Cue, DivergenceSpan, SpeechRegion, Word,
)
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


def _sync(tmp_path, srt: str, words: list[tuple], responses: dict[str, dict[str, object]]):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": word[0], "start": word[1], "end": word[2], "confidence": None,
         **({"speaker_id": word[3]} if len(word) > 3 else {})}
        for word in words
    ]}, ensure_ascii=False), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(fixture)}, "llm": {"provider": "fixture", "responses": responses},
    }, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"
    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.5),
    )
    return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"]


def _decide(case_id: str, final_text: str, verdict: str = "use_audio") -> dict[str, object]:
    return {"case_id": case_id, "verdict": verdict, "final_text": final_text, "confidence": 0.97,
            "reason": "heard in the case audio"}


def _kinds(flags) -> list[str]:
    return [flag["kind"] for flag in flags]


def _near(value_ms: int, seconds: float, tolerance_ms: int = 45) -> bool:
    return abs(value_ms - round(seconds * 1000)) <= tolerance_ms


_NEW_YEAR_SRT = (
    "1\n00:00:01,000 --> 00:00:01,600\nRealizem seus desejos\n\n"
    "2\n00:00:01,640 --> 00:00:02,670\nno Natal.\n\n"
    "3\n00:00:17,000 --> 00:00:18,500\nQuem está falando agora?\n"
)
_NEW_YEAR_WORDS = [
    ("Realizem", 0.72, 1.00), ("seus", 1.02, 1.20), ("desejos", 1.22, 1.46),
    ("no", 1.68, 1.76), ("ano", 1.84, 2.02), ("novo.", 2.12, 2.48),
    ("Alô?", 15.90, 16.30),
    ("Quem", 17.00, 17.20), ("está", 17.22, 17.50), ("falando", 17.52, 18.00), ("agora?", 18.02, 18.40),
]


def test_late_interjection_becomes_its_own_cue_instead_of_being_shown_early(tmp_path):
    # ep11 cue 761: "Alô?" is spoken 14 s after the cue by another actor. It was
    # exported as 'no ano novo. Alô?' at 1.666-2.533 s.
    cues, flags = _sync(tmp_path, _NEW_YEAR_SRT, _NEW_YEAR_WORDS, {"case-1": _decide("case-1", "ano novo. Alô?")})

    assert [cue.plain_text for cue in cues] == [
        "Realizem seus desejos", "no ano novo.", "Alô?", "Quem está falando agora?",
    ]
    edited, interjection = cues[1], cues[2]
    assert _near(edited.start_ms, 1.68) and _near(edited.end_ms, 2.48 + 0.04)
    assert _near(interjection.start_ms, 15.90) and interjection.end_ms <= 17000
    kinds = _kinds(flags)
    assert kinds.count("text_changed") == 1 and kinds.count("adlib_inserted") == 1
    assert "timing_outlier_trimmed" not in kinds and "timing_evidence_held" not in kinds
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_earlier_interjection_of_another_actor_is_not_prefixed_to_the_cue(tmp_path):
    # Scribe ep11 cue 415: "Uhum." (speaker_7) is spoken 4.2 s before the line
    # of speaker_4; the cue was exported as 'Uhum. Vamos lá em cima ...'.
    srt = (
        "1\n00:00:36,030 --> 00:00:37,360\nVocê gosta das nuvens?\n\n"
        "2\n00:00:48,230 --> 00:00:49,600\nVamos lá em cima fazer um pedido.\n"
    )
    words = [
        ("Você", 36.27, 36.50, "speaker_4"), ("gosta", 36.52, 36.80, "speaker_4"),
        ("das", 36.82, 36.95, "speaker_4"), ("nuvens?", 36.97, 37.52, "speaker_4"),
        ("Uhum.", 42.44, 44.08, "speaker_7"),
        ("Vamo", 48.32, 48.44, "speaker_4"), ("lá", 48.46, 48.54, "speaker_4"), ("em", 48.56, 48.62, "speaker_4"),
        ("cima", 48.64, 48.88, "speaker_4"), ("fazer", 48.92, 49.10, "speaker_4"), ("um", 49.12, 49.18, "speaker_4"),
        ("pedido.", 49.20, 49.78, "speaker_4"),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Uhum. Vamos", "hybrid")})

    assert [cue.plain_text for cue in cues] == [
        "Você gosta das nuvens?", "Uhum.", "Vamos lá em cima fazer um pedido.",
    ]
    assert _near(cues[1].start_ms, 42.44)
    assert _near(cues[2].start_ms, 48.32) and _near(cues[2].end_ms, 49.82)
    kinds = _kinds(flags)
    assert kinds.count("adlib_inserted") == 1
    # The cue itself keeps its source wording: nothing to report for it.
    assert "text_changed" not in kinds and "timing_outlier_trimmed" not in kinds
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_whole_cue_keeps_the_group_spoken_at_its_own_time(tmp_path):
    # ep11 MAI cue 114: "Hã?" 12 s before and "Ei," 4.5 s after the name were
    # part of one case; the complete edit was held with two flags.
    srt = (
        "1\n00:00:16,280 --> 00:00:17,030\nOlha.\n\n"
        "2\n00:00:29,960 --> 00:00:30,880\nLuan Nian!\n\n"
        "3\n00:00:35,880 --> 00:00:36,960\nO chefe me procurou.\n"
    )
    words = [
        ("Olha.", 16.64, 16.96),
        ("Hã?", 17.44, 17.72), ("Luanian!", 29.96, 30.72), ("Ei,", 35.20, 35.58),
        ("O", 36.00, 36.06), ("chefe", 36.16, 36.38), ("me", 36.44, 36.52), ("procurou.", 36.56, 36.90),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Hã? Luan Nian! Ei,", "hybrid")})

    assert [cue.plain_text for cue in cues] == ["Olha.", "Hã?", "Luan Nian!", "Ei,", "O chefe me procurou."]
    assert _near(cues[1].start_ms, 17.44)
    assert _near(cues[2].start_ms, 29.96) and _near(cues[2].end_ms, 30.76)
    assert _near(cues[3].start_ms, 35.20)
    kinds = _kinds(flags)
    assert kinds.count("adlib_inserted") == 2
    assert "adjudication_word_mapping_held" not in kinds and "adjudication_replacement_ownership_held" not in kinds
    assert not [flag for flag in flags if flag["severity"] == "error"]


def test_far_word_the_adjudicator_left_out_is_not_inserted(tmp_path):
    # webjob-83a660 cue 48: 'Jawohl!' is spoken "Ja, Frau."; "Oh," 2.7 s later
    # belongs to the next line and was not approved for this cue.
    srt = (
        "1\n00:00:13,630 --> 00:00:15,230\nzum Gebären geboren!\n\n"
        "2\n00:00:15,230 --> 00:00:16,100\nJawohl!\n\n"
        "3\n00:00:18,400 --> 00:00:20,500\nOh! Das ist gut.\n"
    )
    words = [
        ("zum", 14.00, 14.20), ("Gebären", 14.22, 14.70), ("geboren!", 14.72, 15.18),
        ("Ja,", 15.36, 15.50), ("Frau.", 15.60, 15.82), ("Oh,", 18.52, 18.86),
        ("das", 19.10, 19.30), ("ist", 19.32, 19.50), ("gut.", 19.52, 19.90),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Ja, Frau")})

    assert [cue.plain_text for cue in cues][:2] == ["zum Gebären geboren!", "Ja, Frau!"]
    assert len(cues) == 3
    assert _near(cues[1].start_ms, 15.36) and cues[1].end_ms <= 16400
    assert "adlib_inserted" not in _kinds(flags) and "timing_outlier_trimmed" not in _kinds(flags)


def test_far_groups_of_one_insertion_are_placed_separately(tmp_path):
    # Scribe ep11 case-174: "Hum." directly after one line and "Hum," 1.8 s
    # later were prefixed together to the following cue, shown from 1376.30.
    srt = (
        "1\n00:00:14,360 --> 00:00:16,120\nVou esperar o cliente responder.\n\n"
        "2\n00:00:19,120 --> 00:00:20,120\nNão precisa esperar.\n"
    )
    words = [
        ("Vou", 14.30, 14.50, "speaker_4"), ("esperar", 14.52, 14.90, "speaker_4"), ("o", 14.92, 14.98, "speaker_4"),
        ("cliente", 15.00, 15.40, "speaker_4"), ("responder.", 15.42, 16.00, "speaker_4"),
        ("Hum.", 16.60, 16.80, "speaker_4"), ("Hum,", 18.94, 19.09, "speaker_4"),
        ("Não", 19.18, 19.28, "speaker_4"), ("precisa", 19.30, 19.58, "speaker_4"), ("esperar.", 19.62, 19.92, "speaker_4"),
    ]

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "Hum. Hum,")})

    assert [cue.plain_text for cue in cues] == [
        "Vou esperar o cliente responder.", "Hum.", "Hum, Não precisa esperar.",
    ]
    assert _near(cues[1].start_ms, 16.60)
    assert _near(cues[2].start_ms, 18.94)
    assert "timing_outlier_trimmed" not in _kinds(flags)


def test_wording_that_cannot_be_divided_between_far_groups_is_held_with_one_flag(tmp_path):
    # The approved wording shares no word with what the ASR heard, so nothing
    # tells which part of it was spoken 14 s later.
    cues, flags = _sync(tmp_path, _NEW_YEAR_SRT, _NEW_YEAR_WORDS,
                        {"case-1": _decide("case-1", "fim de tudo isso", "hybrid")})

    assert [cue.plain_text for cue in cues] == ["Realizem seus desejos", "no Natal.", "Quem está falando agora?"]
    assert (cues[1].start_ms, cues[1].end_ms) == (1640, 2670)
    kinds = _kinds(flags)
    assert "text_changed" not in kinds and "adlib_inserted" not in kinds
    assert kinds.count("adjudication_replacement_ownership_held") == 1


def _span(**overrides) -> DivergenceSpan:
    values = dict(
        case_id="case-1", cue_ids=[2], srt_text="Natal", asr_text="ano novo. Alô?",
        srt_token_indices=[4], asr_word_indices=[4, 5, 6], start=1.84, end=16.30,
        left_anchor_cue_id=2, right_anchor_cue_id=3, left_anchor_end=1.76, right_anchor_start=17.0,
    )
    values.update(overrides)
    return DivergenceSpan(**values)


def _separate(span: DivergenceSpan, final_text: str, verdict: str = "use_audio"):
    cues = [
        Cue(index=1, start_ms=1000, end_ms=1600, lines=["Realizem seus desejos"]),
        Cue(index=2, start_ms=1640, end_ms=2670, lines=["no Natal."]),
        Cue(index=3, start_ms=17000, end_ms=18500, lines=["Quem está falando agora?"]),
    ]
    words = [Word(text=text, start=start, end=end, confidence=None) for text, start, end in _NEW_YEAR_WORDS]
    alignment = AlignmentResult(
        divergence_spans=[span],
        cue_word_indices={1: [0, 1, 2], 2: [3], 3: [7, 8, 9, 10]},
    )
    decision = AdjudicationDecision(case_id=span.case_id, verdict=verdict, final_text=final_text,
                                    confidence=0.95, reason="heard")
    return separate_detached_speech(cues, alignment, [decision], words, max_intra_cue_gap=1.5)


def test_detached_group_is_a_pure_insertion_anchored_between_its_neighbours():
    alignment, decisions, flags = _separate(_span(), "Ano-Novo. Alô?", "hybrid")

    home, detached = alignment.divergence_spans
    assert flags == []
    assert (home.case_id, home.cue_ids, home.asr_word_indices, home.asr_text) == ("case-1", [2], [4, 5], "ano novo.")
    assert (home.start, home.end, home.srt_token_indices) == (1.84, 2.48, [4])
    assert detached.case_id.startswith(DETACHED_SPEECH_PREFIX) and detached.case_id.endswith("case-1")
    assert (detached.cue_ids, detached.srt_token_indices, detached.srt_text) == ([], [], "")
    assert (detached.asr_word_indices, detached.asr_text, detached.start, detached.end) == ([6], "Alô?", 15.90, 16.30)
    assert (detached.left_anchor_cue_id, detached.left_anchor_end) == (2, 2.48)
    assert (detached.right_anchor_cue_id, detached.right_anchor_start) == (3, 17.0)
    assert [(decision.case_id, decision.verdict, decision.final_text) for decision in decisions] == [
        ("case-1", "hybrid", "Ano-Novo."), (detached.case_id, "hybrid", "Alô?"),
    ]


def _police_case(regions: list[SpeechRegion] | None):
    # ep17 cue 326: MAI timed "Ah." at 1250.16 s, 19 s before 'Chama a polícia.'
    cues = [
        Cue(index=325, start_ms=1_247_600, end_ms=1_249_360, lines=["Eu tenho vontade de te matar agora."]),
        Cue(index=326, start_ms=1_269_600, end_ms=1_270_430, lines=["Chamar a polícia."]),
    ]
    words = [Word(text=text, start=start, end=end, confidence=None) for text, start, end in [
        ("agora.", 1248.72, 1249.04), ("Ah.", 1250.16, 1250.72), ("Chama", 1269.72, 1269.92),
        ("a", 1269.94, 1269.98), ("polícia.", 1270.04, 1270.56),
    ]]
    span = DivergenceSpan(
        case_id="case-136", cue_ids=[326], srt_text="Chamar", asr_text="Ah. Chama",
        srt_token_indices=[7], asr_word_indices=[1, 2], start=1250.16, end=1269.92,
        left_anchor_cue_id=325, right_anchor_cue_id=326, left_anchor_end=1249.04, right_anchor_start=1269.94,
    )
    alignment = AlignmentResult(divergence_spans=[span], cue_word_indices={325: [0], 326: [3, 4]})
    decision = AdjudicationDecision(case_id="case-136", verdict="use_audio", final_text="Ah. Chama",
                                    confidence=0.95, reason="heard")
    return separate_detached_speech(cues, alignment, [decision], words, speech_regions=regions)


def test_far_interjection_stays_with_the_cue_when_unlabelled_speech_adjoins_the_cue():
    # A burst without any ASR word ends 0.25 s before "Chama": the reviewer
    # kept 'Ah. Chama a polícia.' and started the cue with that burst.
    regions = [SpeechRegion(start=1247.66, end=1249.03), SpeechRegion(start=1250.04, end=1250.70),
               SpeechRegion(start=1269.15, end=1269.47), SpeechRegion(start=1269.60, end=1270.43)]

    alignment, decisions, flags = _police_case(regions)

    (home,) = alignment.divergence_spans
    assert (home.asr_word_indices, home.asr_text, home.start) == ([2], "Chama", 1269.72)
    assert [(decision.case_id, decision.final_text) for decision in decisions] == [("case-136", "Ah. Chama")]
    assert flags == []


def test_far_interjection_is_detached_when_only_its_own_time_has_speech():
    regions = [SpeechRegion(start=1247.66, end=1249.03), SpeechRegion(start=1250.04, end=1250.70),
               SpeechRegion(start=1269.60, end=1270.43)]

    for speech_regions in (regions, None):
        alignment, decisions, _ = _police_case(speech_regions)
        detached, home = alignment.divergence_spans
        assert (detached.cue_ids, detached.asr_word_indices, detached.right_anchor_cue_id) == ([], [1], 326)
        assert [decision.final_text for decision in decisions] == ["Ah.", "Chama"]
        assert home.asr_word_indices == [2]


def test_far_word_running_into_the_next_line_is_not_kept_with_the_earlier_cue():
    # ep11 MAI cue 330: "Eu" is spoken 5.6 s later, 20 ms before the next
    # cue's words. A noise right after the cue must not keep it there.
    cues = [
        Cue(index=330, start_ms=1_002_750, end_ms=1_003_920, lines=["Lá em cima não tem banheiro."]),
        Cue(index=331, start_ms=1_009_230, end_ms=1_010_080, lines=["Vou segurar mais um pouco."]),
    ]
    words = [Word(text=text, start=start, end=end, confidence=None) for text, start, end in [
        ("Tá", 1002.92, 1003.02), ("bom,", 1003.04, 1003.18), ("então", 1003.22, 1003.38),
        ("vamos", 1003.42, 1003.56), ("lá.", 1003.60, 1003.76), ("Eu", 1009.40, 1009.46),
        ("vou", 1009.48, 1009.60), ("esperar", 1009.62, 1009.90),
    ]]
    span = DivergenceSpan(
        case_id="case-117", cue_ids=[330], srt_text="Lá em cima não tem banheiro",
        asr_text="Tá bom, então vamos lá. Eu", srt_token_indices=[0, 1, 2, 3, 4, 5],
        asr_word_indices=[0, 1, 2, 3, 4, 5], start=1002.92, end=1009.46,
        right_anchor_cue_id=331, right_anchor_start=1009.48,
    )
    alignment = AlignmentResult(divergence_spans=[span], cue_word_indices={331: [6, 7]})
    decision = AdjudicationDecision(case_id="case-117", verdict="use_audio", final_text="Tá bom, então vamos lá. Eu",
                                    confidence=0.95, reason="heard")
    regions = [SpeechRegion(start=1002.90, end=1003.80), SpeechRegion(start=1004.00, end=1004.30),
               SpeechRegion(start=1009.35, end=1010.10)]

    alignment, decisions, _ = separate_detached_speech(cues, alignment, [decision], words, speech_regions=regions)

    home, detached = alignment.divergence_spans
    assert [decision.final_text for decision in decisions] == ["Tá bom, então vamos lá.", "Eu"]
    assert (home.asr_word_indices, detached.asr_word_indices) == ([0, 1, 2, 3, 4], [5])
    assert (detached.left_anchor_cue_id, detached.right_anchor_cue_id, detached.right_anchor_start) == (330, 331, 1009.48)


_ONLY_FAR_WORDS = [
    ("Realizem", 0.72, 1.00), ("seus", 1.02, 1.20), ("desejos", 1.22, 1.46), ("no", 1.68, 1.76),
    ("Alô?", 15.90, 16.30),
    ("Quem", 17.00, 17.20), ("está", 17.22, 17.50), ("falando", 17.52, 18.00), ("agora?", 18.02, 18.40),
]


def test_single_approved_word_spoken_far_from_the_cue_is_not_written_into_it(tmp_path):
    # The case's only word is spoken 14 s after the cue's retained "no": the
    # approved wording replaced "Natal" inside the cue, shown 14 s early.
    # 'Natal' is now asked about at the cue's own time, "Alô?" at its own.
    srt = _NEW_YEAR_SRT.replace("00:00:02,670", "00:00:02,400")
    cues, flags = _sync(tmp_path, srt, _ONLY_FAR_WORDS, {
        "case-1": _decide("case-1", "Alô?"),
        f"{DETACHED_SPEECH_PREFIX}tail-case-1": _decide(f"{DETACHED_SPEECH_PREFIX}tail-case-1", ""),
    })

    assert [cue.plain_text for cue in cues] == ["Realizem seus desejos", "no.", "Alô?", "Quem está falando agora?"]
    assert _near(cues[1].start_ms, 1.68) and _near(cues[2].start_ms, 15.90)
    kinds = _kinds(flags)
    assert kinds.count("text_changed") == 1 and kinds.count("adlib_inserted") == 1
    assert "timing_outlier_trimmed" not in kinds and not [flag for flag in flags if flag["severity"] == "error"]


def test_far_word_directly_before_the_next_line_still_joins_that_line(tmp_path):
    # ep17 case-99: 'de vez' was not spoken; "E" is spoken 4 s later, 80 ms
    # before the next cue's words.
    srt = (
        "1\n00:00:07,670 --> 00:00:09,240\neu vou estar ferrada de vez.\n\n"
        "2\n00:00:12,950 --> 00:00:14,950\nSe não for só eu a vítima,\n"
    )
    words = [
        ("eu", 8.00, 8.10), ("vou", 8.12, 8.24), ("estar", 8.26, 8.50), ("ferrada.", 8.52, 8.92),
        ("E", 13.04, 13.12),
        ("se", 13.20, 13.30), ("não", 13.32, 13.50), ("for", 13.52, 13.70), ("só", 13.72, 13.84), ("eu", 13.86, 13.96),
        ("a", 13.98, 14.02), ("vítima,", 14.06, 14.60),
    ]

    cues, _ = _sync(tmp_path, srt, words, {
        "case-1": _decide("case-1", "E"),
        f"{DETACHED_SPEECH_PREFIX}tail-case-1": _decide(f"{DETACHED_SPEECH_PREFIX}tail-case-1", ""),
    })

    assert [cue.plain_text for cue in cues] == ["eu vou estar ferrada.", "E se não for só eu a vítima,"]
    assert _near(cues[0].end_ms, 8.96) and _near(cues[1].start_ms, 13.04)


def test_far_wording_with_other_words_than_the_audio_is_not_placed():
    # The approved text keeps a source word: it is not the far speech alone.
    # (An aligner case asks about 'Natal' and "Alô?" apart before hearing; a
    # derived question is still divided only after it.)
    far_only = _span(asr_text="Alô?", asr_word_indices=[6], start=15.90, end=16.30)
    alignment, decisions, flags = _separate(far_only, "Natal, alô?", "hybrid")

    assert alignment.divergence_spans == [far_only] and flags == []
    assert [(decision.case_id, decision.final_text) for decision in decisions] == [("case-1", "Natal, alô?")]


def test_single_far_word_does_not_time_the_kept_cue(tmp_path):
    # ep17 cue 140: 'Vou voltar primeiro.' is kept; the case's only ASR word is
    # "Ah!" 5.3 s later. It was assigned to the cue and reported as a trimmed outlier.
    srt = (
        "1\n00:00:05,110 --> 00:00:05,870\nVou voltar primeiro.\n\n"
        "2\n00:00:13,000 --> 00:00:14,000\nAté amanhã então.\n"
    )
    words = [
        ("Vou", 5.255, 5.379), ("voltar.", 5.480, 5.715), ("Ah!", 11.245, 11.435),
        ("Até", 13.00, 13.20), ("amanhã", 13.22, 13.60), ("então.", 13.62, 13.95),
    ]

    tail = f"{DETACHED_SPEECH_PREFIX}tail-case-1"
    cues, flags = _sync(tmp_path, srt, words, {tail: _decide(tail, "primeiro", "keep_srt")})

    assert [cue.plain_text for cue in cues] == ["Vou voltar primeiro.", "Até amanhã então."]
    assert _near(cues[0].start_ms, 5.255) and cues[0].end_ms < 7000
    assert "timing_outlier_trimmed" not in _kinds(flags)


def test_span_without_a_large_gap_is_left_alone():
    near = _span(asr_text="ano novo.", asr_word_indices=[4, 5], end=2.48)
    alignment, decisions, flags = _separate(near, "ano novo.")
    assert alignment.divergence_spans == [near] and decisions[0].final_text == "ano novo." and flags == []


def test_kept_source_text_is_timed_only_by_the_words_at_its_own_time(tmp_path):
    alignment, decisions, flags = _separate(_span(), "Natal", "keep_srt")
    (home,) = alignment.divergence_spans
    assert (home.case_id, home.asr_word_indices, home.asr_text, home.end) == ("case-1", [4, 5], "ano novo.", 2.48)
    assert [(decision.verdict, decision.final_text) for decision in decisions] == [("keep_srt", "Natal")] and flags == []

    # The far word was assigned to the kept cue and reported as a trimmed outlier.
    cues, report_flags = _sync(tmp_path, _NEW_YEAR_SRT, _NEW_YEAR_WORDS, {"case-1": _decide("case-1", "Natal", "keep_srt")})
    assert [cue.plain_text for cue in cues] == ["Realizem seus desejos", "no Natal.", "Quem está falando agora?"]
    assert _near(cues[1].start_ms, 1.68) and _near(cues[1].end_ms, 2.52)
    assert "timing_outlier_trimmed" not in _kinds(report_flags)


def _aligned(srt: str, words: list[tuple]):
    cues = parse_srt_text(srt)
    spoken = [Word(text=text, start=start, end=end, confidence=None) for text, start, end in words]
    return cues, spoken, align_cues_to_words(cues, spoken).divergence_spans


_FAR_TAIL_SRT = (
    "1\n00:00:05,110 --> 00:00:05,870\nVou voltar primeiro.\n\n"
    "2\n00:00:12,520 --> 00:00:13,400\nO que você quer?\n"
)
# ep17 cue 140 shape: 'primeiro' is not spoken; "Ah!" is spoken 3 s after 'voltar.'.
_FAR_TAIL_WORDS = [
    ("Vou", 5.24, 5.38), ("voltar.", 5.48, 5.88), ("Ah!", 8.88, 9.08),
    ("O", 12.64, 12.68), ("que", 12.72, 12.82), ("você", 12.86, 13.10), ("quer?", 13.12, 13.40),
]


def test_cue_tail_and_far_word_are_asked_about_at_their_own_times():
    cues, words, spans = _aligned(_FAR_TAIL_SRT, _FAR_TAIL_WORDS)
    (shared,) = spans
    assert (shared.cue_ids, shared.srt_text, shared.asr_text) == ([1], "primeiro", "Ah!")

    tail, far = separate_unheard_cue_edges(spans, cues, words, max_intra_cue_gap=1.5)

    assert tail.case_id == f"{DETACHED_SPEECH_PREFIX}tail-case-1"
    assert (tail.cue_ids, tail.srt_text, tail.srt_token_indices) == ([1], "primeiro", shared.srt_token_indices)
    assert (tail.asr_word_indices, tail.asr_text, tail.start, tail.end) == ([], "", 5.88, 5.88 + 1.5)
    assert (tail.left_anchor_cue_id, tail.left_anchor_end, tail.right_anchor_cue_id, tail.right_anchor_start) == (
        1, 5.88, None, None)
    assert (far.case_id, far.cue_ids, far.srt_text, far.srt_token_indices) == ("case-1", [], "", [])
    assert (far.asr_word_indices, far.asr_text, far.start, far.end) == ([2], "Ah!", 8.88, 9.08)
    assert (far.left_anchor_cue_id, far.right_anchor_cue_id, far.insertion_token_offset) == (1, 2, None)


def test_retained_cue_edge_far_from_the_words_of_a_shared_case_is_asked_alone():
    # ep17 case-212: 'do seu prédio' ends cue 527; the case's only word "Sem"
    # is spoken 11 s later, directly before the next cue's retained words.
    srt = (
        "1\n00:00:10,000 --> 00:00:11,300\nEu moro no último andar do seu prédio.\n\n"
        "2\n00:00:21,700 --> 00:00:22,900\nNão tenho medo de altura.\n"
    )
    cues, words, spans = _aligned(srt, [
        ("Eu", 10.00, 10.10), ("moro", 10.12, 10.30), ("no", 10.32, 10.40), ("último", 10.42, 10.70),
        ("andar.", 10.72, 11.00), ("Sem", 21.82, 22.02),
        ("medo", 22.12, 22.30), ("de", 22.32, 22.38), ("altura.", 22.40, 22.80),
    ])
    (shared,) = spans
    assert (shared.cue_ids, shared.asr_text) == ([1, 2], "Sem")

    tail, rest = separate_unheard_cue_edges(spans, cues, words, max_intra_cue_gap=1.5)

    assert (tail.case_id, tail.cue_ids, tail.srt_text, tail.asr_word_indices) == (
        f"{DETACHED_SPEECH_PREFIX}tail-case-1", [1], "do seu prédio", [])
    assert (tail.start, tail.end) == (11.00, 11.00 + 1.5)
    assert (rest.case_id, rest.cue_ids, rest.srt_text, rest.asr_text) == ("case-1", [2], "Não tenho", "Sem")
    assert (rest.start, rest.end, rest.right_anchor_cue_id) == (21.82, 22.02, 2)


def test_words_near_the_cue_or_inside_its_own_pause_keep_the_shared_case():
    near = [*_FAR_TAIL_WORDS[:2], ("primeira.", 6.10, 6.50), *_FAR_TAIL_WORDS[3:]]
    cues, words, spans = _aligned(_FAR_TAIL_SRT, near)
    assert separate_unheard_cue_edges(spans, cues, words, max_intra_cue_gap=1.5) == spans

    # The cue's own retained words surround the far word: the pause is its own.
    cues, words, spans = _aligned("1\n00:00:01,000 --> 00:00:07,500\nEu vou agora mesmo embora.\n", [
        ("Eu", 1.00, 1.20), ("vou", 1.22, 1.40), ("Hã?", 4.00, 4.20), ("mesmo", 6.60, 6.90), ("embora.", 6.92, 7.40),
    ])
    assert [span.asr_text for span in spans] == ["Hã?"]
    assert separate_unheard_cue_edges(spans, cues, words, max_intra_cue_gap=1.5) == spans


class _ClipRecordingAdapter:
    """Records each question with its clip; answers like a reviewer who heard it."""

    def __init__(self, approve: bool):
        self.approve = approve
        self.questions: list[tuple[DivergenceSpan, AudioSnippet]] = []

    def adjudicate(self, spans):
        raise AssertionError("a required-audio case must not be answered from text")

    def adjudicate_with_audio(self, spans, snippets):
        self.questions.extend((span, snippets[span.case_id]) for span in spans)
        if not self.approve:
            return []
        # The cue tail is not heard at the cue's time; the far word is heard at its own.
        return [{"case_id": span.case_id, "verdict": "use_audio", "final_text": span.asr_text, "confidence": 0.95,
                 "evidence": "heard_clearly", "heard_text": span.asr_text, "reason": "heard in the clip"}
                for span in spans]


def _silent_wav(path, seconds: float):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _hearing_run(tmp_path, monkeypatch, srt: str, words: list[tuple], adapter: _ClipRecordingAdapter):
    # Clip windows come from the real window rule; only ffmpeg is replaced.
    def extract(_audio, spans, directory, *, pad_seconds, max_duration_seconds, max_covering_duration_seconds,
                **_kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        snippets = []
        for span in spans:
            start, end = _snippet_window(
                span.start, span.end, pad_seconds, max_duration_seconds, max_covering_duration_seconds,
            )
            path = _silent_wav(directory / f"{span.case_id}.wav", 0.1)
            snippets.append(AudioSnippet(
                case_id=span.case_id, path=str(path), start=round(start, 3), end=round(end, 3),
            ))
        return snippets

    original = pipeline.llm_adapter_from_config
    monkeypatch.setattr(pipeline, "extract_audio_snippets", extract)
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda config, pass_name=None: (
        adapter if pass_name == "adjudication" else original(config, pass_name=pass_name)))
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = _silent_wav(tmp_path / "episode.wav", 15)
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end, "confidence": None} for text, start, end in words
    ]}, ensure_ascii=False), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(fixture)},
        "llm": {"provider": "fixture", "responses": {}, "adjudication": {
            "audio_snippet_double_check": {"enabled": True, "pad_seconds": 2.0, "max_duration_seconds": 20.0},
        }},
    }, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"

    def run(**kwargs):
        result = pipeline.sync_episode(
            source, audio, output, tmp_path / "work", providers_path=providers,
            style_profile=StyleProfile(fps=30, min_cue_dur=0.5), **kwargs,
        )
        return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"]

    return run


def _contains(snippet: AudioSnippet, start: float, end: float) -> bool:
    return snippet.start <= start and snippet.end >= end


def test_cue_tail_is_heard_at_its_own_time_and_an_unanswered_far_word_is_flagged(tmp_path, monkeypatch):
    # The clip of the shared case was cut around "Ah!" and did not contain the
    # cue; the reviewer kept 'primeiro' unheard and "Ah!" ended in no cue and no flag.
    adapter = _ClipRecordingAdapter(approve=False)
    cues, flags = _hearing_run(tmp_path, monkeypatch, _FAR_TAIL_SRT, _FAR_TAIL_WORDS, adapter)()

    asked = {span.case_id: (span, clip) for span, clip in adapter.questions}
    assert sorted(asked) == ["case-1", f"{DETACHED_SPEECH_PREFIX}tail-case-1"]
    tail, tail_clip = asked[f"{DETACHED_SPEECH_PREFIX}tail-case-1"]
    far, far_clip = asked["case-1"]
    assert (tail.cue_ids, tail.srt_text, tail.asr_text) == ([1], "primeiro", "")
    assert _contains(tail_clip, 5.11, 5.88)  # the cue as written and as spoken
    assert (far.cue_ids, far.asr_text) == ([], "Ah!") and _contains(far_clip, 8.88, 9.08)

    assert [cue.plain_text for cue in cues] == ["Vou voltar primeiro.", "O que você quer?"]
    assert _near(cues[0].start_ms, 5.24) and cues[0].end_ms < 7000
    assert [flag["kind"] for flag in flags if flag["cue_ids"] == [1]] == ["invalid_llm_response"]
    # The far word is held for review at its own time.
    assert [flag["kind"] for flag in flags if not flag["cue_ids"] and flag["start"] is not None
            and abs(flag["start"] - 8.88) < 0.01 and abs(flag["end"] - 9.08) < 0.01] == ["invalid_llm_response"]


def test_approved_cue_tail_deletion_and_far_word_place_the_word_once(tmp_path, monkeypatch):
    adapter = _ClipRecordingAdapter(approve=True)
    cues, flags = _hearing_run(tmp_path, monkeypatch, _FAR_TAIL_SRT, _FAR_TAIL_WORDS, adapter)()

    assert len(adapter.questions) == 2
    assert [cue.plain_text for cue in cues] == ["Vou voltar.", "Ah!", "O que você quer?"]
    assert _near(cues[0].start_ms, 5.24) and _near(cues[1].start_ms, 8.88)
    assert sum(cue.plain_text.count("Ah!") for cue in cues) == 1
    kinds = _kinds(flags)
    assert kinds.count("text_changed") == 1 and kinds.count("adlib_inserted") == 1
    assert not [flag for flag in flags if flag["severity"] == "error"]


@pytest.mark.parametrize("mode", ["rebuild", "verify"])
def test_answer_heard_away_from_the_cue_cannot_resume_without_its_own_questions(tmp_path, monkeypatch, mode):
    adapter = _ClipRecordingAdapter(approve=True)
    run = _hearing_run(tmp_path, monkeypatch, _FAR_TAIL_SRT, _FAR_TAIL_WORDS, adapter)
    with monkeypatch.context() as legacy:
        legacy.setattr(pipeline, "_alignment_with_unheard_cue_edges", lambda alignment, *_a, **_k: alignment)
        run()
    adapter.questions.clear()

    with pytest.raises(ValueError, match="resume from adjudicate"):
        run(resume=mode)
    assert adapter.questions == []
    cues, _ = run(resume="adjudicate")
    assert sorted(span.case_id for span, _ in adapter.questions) == ["case-1", f"{DETACHED_SPEECH_PREFIX}tail-case-1"]
    assert [cue.plain_text for cue in cues] == ["Vou voltar.", "Ah!", "O que você quer?"]
