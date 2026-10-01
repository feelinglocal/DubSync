"""Words spoken seconds away from a cue are never shown inside that cue."""
from __future__ import annotations

import json

import yaml

from dubsync import pipeline
from dubsync.detached_speech import DETACHED_SPEECH_PREFIX, separate_detached_speech
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, SpeechRegion, Word
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

    cues, flags = _sync(tmp_path, srt, words, {"case-1": _decide("case-1", "primeiro", "keep_srt")})

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
