from __future__ import annotations

import json
import socket
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.asr_timing import repair_asr_word_edges
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, SpeechRegion, TokenMatch, Word
from dubsync.output_order import finalize_cues_for_output
from dubsync.recue import cue_spoken_spans, rebuild_cues, select_cue_word_window
from dubsync.style_profile import StyleProfile
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.timing_refinement import refine_cues_to_speech_activity


def _heard_interjection():
    """Native Scribe 1B source59: the retained final word outlasts VAD by 65 ms."""
    cues = [
        Cue(index=59, start_ms=113360, end_ms=115330, lines=["うあっ！"]),
        Cue(index=60, start_ms=116130, end_ms=116930, lines=["放したぞ"]),
    ]
    words = [Word(text=text, start=start, end=end) for text, start, end in [
        ("う", 113.035, 113.200), ("わ", 113.240, 113.340),
        ("離", 115.620, 115.680), ("し", 115.680, 115.760),
        ("た", 115.760, 115.840), ("ぞ", 115.860, 115.940),
    ]]
    alignment = AlignmentResult(
        cue_word_indices={59: [0, 1], 60: [2, 3, 4, 5]},
        token_matches=[TokenMatch(cue_id=cue, srt_token_index=source, asr_word_index=word, score=1)
                       for cue, source, word in [(59, 0, 0), (60, 4, 3), (60, 5, 4), (60, 6, 5)]],
        divergence_spans=[DivergenceSpan(
            case_id="case-29", cue_ids=[59, 60], srt_text="あっ放", asr_text="わ離",
            srt_token_indices=[1, 2, 3], asr_word_indices=[1, 2],
        )],
    )
    return dict(
        cues=cues, source_cues=list(cues), words=words, alignment=alignment,
        decisions=[AdjudicationDecision(
            case_id="case-29", verdict="keep_srt", final_text="あっ放", confidence=1,
            evidence="heard_clearly", heard_text="あっ放", reason="Native reply confirms the source fragment.",
        )],
        regions=[SpeechRegion(start=113.035, end=113.275), SpeechRegion(start=115.365, end=116.095)],
        protected_cue_ids=set(),
    )


def test_native_heard_interjection_uses_uniquely_owned_retained_word_tail():
    case = _heard_interjection()
    repaired, _ = repair_asr_word_edges(case["words"], case["regions"])
    assert repaired[1] == case["words"][1]
    confirmed = pipeline._confirmed_source_wording_cue_ids(**case)
    assert 59 in confirmed
    rebuilt, flags = rebuild_cues(
        case["cues"], case["words"], case["alignment"], StyleProfile(fps=30, min_cue_dur=.5),
        confirmed_wording_cue_ids=confirmed,
    )
    assert rebuilt[0].start_ms <= 113035
    assert rebuilt[0].end_ms >= 113340
    assert not any(flag.kind == "timing_evidence_held" and 59 in flag.cue_ids for flag in flags)


@pytest.mark.parametrize("fault", [
    "no_overlap", "long_tail", "second_burst", "foreign_word", "foreign_tail", "owned_foreign_tail", "shared_word", "no_anchor",
])
def test_native_heard_fragment_does_not_release_unowned_timing(fault):
    case = _heard_interjection()
    if fault == "no_overlap":
        case["words"][1] = case["words"][1].model_copy(update={"start": 113.280})
    elif fault == "long_tail":
        case["words"][1] = case["words"][1].model_copy(update={"end": 113.600})
    elif fault == "second_burst":
        case["regions"].append(SpeechRegion(start=113.300, end=113.340))
    elif fault == "foreign_word":
        case["words"].append(Word(text="hey", start=113.245, end=113.265))
    elif fault in {"foreign_tail", "owned_foreign_tail"}:
        case["words"].append(Word(text="hey", start=113.280, end=113.330))
        if fault == "owned_foreign_tail":
            case["alignment"].cue_word_indices[60].append(len(case["words"]) - 1)
    elif fault == "shared_word":
        case["alignment"].cue_word_indices[60].append(1)
    elif fault == "no_anchor":
        case["alignment"].token_matches = [match for match in case["alignment"].token_matches if match.cue_id != 59]
    assert 59 not in pipeline._confirmed_source_wording_cue_ids(**case)


def test_native_heard_tail_respects_the_configured_overrun_limit():
    assert 59 not in pipeline._confirmed_source_wording_cue_ids(**_heard_interjection(), max_region_overrun=.05)


def test_adjacent_word_after_retained_tail_does_not_share_the_utterance():
    case = _heard_interjection()
    case["words"].append(Word(text="hey", start=113.340, end=113.430))
    assert 59 in pipeline._confirmed_source_wording_cue_ids(**case)


@pytest.mark.parametrize("resume", [None, "rebuild", "verify"])
def test_retained_interjection_tail_reaches_fresh_and_resumed_output(tmp_path, monkeypatch, resume):
    def no_network(*unused, **kwargs):
        raise AssertionError("Captured timing regression must stay offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    case = _heard_interjection()
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000 * 6)
    cues = [cue.with_timing(cue.start_ms - 112000, cue.end_ms - 112000) for cue in case["cues"]]
    words = [word.model_copy(update={"start": word.start - 112, "end": word.end - 112}) for word in case["words"]]
    regions = [region.model_copy(update={"start": region.start - 112, "end": region.end - 112}) for region in case["regions"]]
    source = tmp_path / "source.srt"
    source.write_text(write_srt(cues), encoding="utf-8")
    word_path = tmp_path / "words.json"
    word_path.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    region_path = tmp_path / "regions.json"
    region_path.write_text(json.dumps({"regions": [region.model_dump() for region in regions]}), encoding="utf-8")
    config = tmp_path / "providers.yaml"
    decision = case["decisions"][0].model_copy(update={"case_id": "case-1"})
    config.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(word_path)},
        "vad": {"fixture_path": str(region_path), "boundary_refinement": True},
        "llm": {"provider": "fixture", "responses": {"case-1": decision.model_dump()}},
    }, allow_unicode=True), encoding="utf-8")
    options = dict(
        srt_path=source, audio_path=audio, output_path=tmp_path / "output.srt",
        workdir=tmp_path / "work", providers_path=config, style_profile=StyleProfile(fps=30, min_cue_dur=.5),
    )
    result = pipeline.sync_episode(**options)
    if resume:
        result = pipeline.sync_episode(**options, resume=resume)
    output = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert output[0].lines == ["うあっ！"]
    assert output[0].start_ms <= 1035
    assert output[0].end_ms >= 1340
    assert not any(flag["kind"] == "timing_evidence_held" and 59 in flag["cue_ids"] for flag in result.report["flags"])


@pytest.mark.parametrize("last_start,last_end,burst_end,rebuilt_end", [
    (90.920, 90.980, 90.925, 91034),
    (90.780, 90.840, 90.805, 90867),
])
def test_refinement_keeps_retained_final_particle_before_a_held_cue(last_start, last_end, burst_end, rebuilt_end):
    words = [Word(text="消すから", start=89.8, end=90.5), Word(text="と", start=last_start, end=last_end)]
    regions = [SpeechRegion(start=89.765, end=burst_end), SpeechRegion(start=91.245, end=91.425)]
    repaired, _ = repair_asr_word_edges(words, regions)
    assert repaired[-1] == words[-1]
    cues = [
        Cue(index=65, start_ms=89767, end_ms=rebuilt_end, lines=["消すからと"]),
        Cue(index=66, start_ms=90733, end_ms=91566, lines=["ははは"]),
    ]
    refined, _ = refine_cues_to_speech_activity(
        cues, regions, StyleProfile(fps=30, min_cue_dur=.5), words=words,
        alignment=AlignmentResult(cue_word_indices={65: [0, 1], 66: []}), protected_cue_ids={66},
    )
    assert refined[0].end_ms >= last_end * 1000
    assert refined[1] == cues[1]


def test_split_short_phrase_preserves_source_timing_instead_of_timing_half_a_word():
    cue = Cue(index=50, start_ms=64666, end_ms=65466, lines=["三分？"])
    words = [Word(text="三", start=64.739, end=64.760), Word(text="分", start=71.140, end=71.160)]
    rebuilt, flags = rebuild_cues(
        [cue], words, AlignmentResult(cue_word_indices={50: [0, 1]}), StyleProfile(fps=30, min_cue_dur=.5),
    )
    assert rebuilt == [cue]
    assert any(flag.kind == "timing_evidence_held" and 50 in flag.cue_ids for flag in flags)


@pytest.mark.parametrize("extra", ["?", "other", "分"])
def test_trimming_an_unneeded_outlier_keeps_the_complete_phrase_timed(extra):
    cue = Cue(index=50, start_ms=64666, end_ms=65466, lines=["三分？"])
    words = [
        Word(text="三", start=64.739, end=64.820), Word(text="分", start=64.830, end=65.000),
        Word(text=extra, start=71.140, end=71.160),
    ]
    rebuilt, flags = rebuild_cues(
        [cue], words, AlignmentResult(cue_word_indices={50: [0, 1, 2]}), StyleProfile(fps=30, min_cue_dur=.5),
    )
    assert rebuilt[0].start_ms <= 64739
    assert 65000 <= rebuilt[0].end_ms < 71000
    assert not any(flag.kind == "timing_evidence_held" for flag in flags)


def test_real_phrase_pause_keeps_both_lexical_groups():
    cue = Cue(index=1, start_ms=500, end_ms=4000, lines=["三分"])
    words = [Word(text="三", start=1.0, end=1.2), Word(text="分", start=2.8, end=3.0)]
    rebuilt, flags = rebuild_cues(
        [cue], words, AlignmentResult(cue_word_indices={1: [0, 1]}), StyleProfile(fps=30, min_cue_dur=.5),
    )
    assert rebuilt[0].start_ms <= 1000
    assert rebuilt[0].end_ms >= 3000
    assert not any(flag.kind == "timing_evidence_held" for flag in flags)


def test_reliable_lexical_group_remains_usable_after_an_impossible_gap():
    cue = Cue(index=1, start_ms=1000, end_ms=5000, lines=["Sim. Vamos."])
    words = [Word(text="Sim.", start=1.0, end=1.3), Word(text="Vamos.", start=9.0, end=9.3)]
    rebuilt, flags = rebuild_cues(
        [cue], words, AlignmentResult(cue_word_indices={1: [0, 1]}), StyleProfile(fps=30, min_cue_dur=.1),
    )
    assert 1300 <= rebuilt[0].end_ms < 1400
    assert not any(flag.kind == "timing_evidence_held" for flag in flags)


@pytest.mark.parametrize("text,word_data,regions,expected_start", [
    ("あのお方は情の深い方で", [
        ("。", 50.560, 50.639), ("あ", 50.735, 50.879), ("の", 50.879, 50.959),
        ("お", 50.960, 51.039), ("方", 51.080, 51.159), ("は", 51.240, 51.319),
        ("情", 51.400, 51.479), ("の", 51.560, 51.639), ("深", 51.680, 51.760),
        ("い", 51.840, 51.919), ("家", 51.960, 52.039), ("庭", 52.040, 52.120), ("で", 52.120, 52.199),
    ], [(48.525, 50.555), (50.735, 53.625)], 50734),
    ("要らないらしいな？", [
        ("、", 89.600, 89.680), ("い", 89.805, 89.960), ("ら", 89.960, 90.039),
        ("な", 90.120, 90.199), ("い", 90.200, 90.279), ("ら", 90.319, 90.399),
        ("し", 90.440, 90.500), ("い", 90.500, 90.599), ("な", 90.640, 90.885),
    ], [(89.035, 89.575), (89.805, 90.885)], 89800),
    ("膨大な権力と", [
        ("、", 47.240, 47.319), ("膨", 47.360, 47.439), ("大", 47.519, 47.599),
        ("な", 47.680, 47.760), ("権", 47.840, 47.919), ("力", 47.960, 48.039), ("と", 48.120, 48.199),
    ], [(45.115, 49.375)], 47334),
    ("天神谷に", [
        ("、", 44.640, 44.745), ("天", 44.825, 44.959), ("神", 45.040, 45.120),
        ("谷", 45.240, 45.319), ("に", 45.480, 45.559),
    ], [(41.675, 44.745), (44.825, 47.405)], 44800),
])
def test_prior_phrase_delimiter_does_not_start_japanese_dialogue_early(text, word_data, regions, expected_start):
    words = [Word(text=token, start=start, end=end) for token, start, end in word_data]
    cue = Cue(index=1, start_ms=int(words[0].start * 1000), end_ms=int(words[-1].end * 1000) + 100, lines=[text])
    alignment = AlignmentResult(cue_word_indices={1: list(range(len(words)))})
    original_ownership = alignment.model_dump()
    profile = StyleProfile(fps=30, min_cue_dur=.5)
    rebuilt, flags = rebuild_cues([cue], words, alignment, profile)
    refined, _ = refine_cues_to_speech_activity(
        rebuilt, [SpeechRegion(start=start, end=end) for start, end in regions], profile,
        words=words, alignment=alignment,
    )
    assert refined[0].start_ms == expected_start
    assert refined[0].lines == cue.lines
    assert alignment.model_dump() == original_ownership
    assert cue_spoken_spans([cue], words, alignment)[1][0] == round(words[1].start * 1000)
    assert not any(flag.kind == "timing_outlier_trimmed" for flag in flags)


def _selected_window(text, tokens):
    cue = Cue(index=1, start_ms=1000, end_ms=2000, lines=[text])
    words = [Word(text=token, start=1.0 + index * .2, end=1.1 + index * .2) for index, token in enumerate(tokens)]
    selected, trimmed = select_cue_word_window(cue, words, max_word_duration=2, max_intra_cue_gap=1.5)
    return words, selected, trimmed


@pytest.mark.parametrize("tokens,expected", [
    (["。", "hello", "!"], ["hello", "!"]),
    (["「", "hello", "world", "。」"], ["hello", "world", "。」"]),
    (["hello", ",", "world"], ["hello", ",", "world"]),
    (["…", "!"], ["…", "!"]),
    (["…"], ["…"]),
])
def test_only_leading_sentence_delimiters_are_excluded_from_spoken_windows(tokens, expected):
    _, selected, trimmed = _selected_window("hello world", tokens)
    assert [word.text for word in selected] == expected
    assert not trimmed


@pytest.mark.parametrize("symbol", ["%", "$", "€", "£", "¥", "+", "-", "−", "=", "&", "@"])
def test_spoken_symbols_keep_their_owned_boundary_timestamps(symbol):
    words, selected, trimmed = _selected_window("five", [symbol, "five", symbol])
    assert selected == words
    assert not trimmed


@pytest.mark.parametrize("delimiter,digit", [(".", "5"), (",", "5"), (".", "five")])
def test_leading_decimal_delimiter_keeps_its_spoken_boundary(delimiter, digit):
    words, selected, trimmed = _selected_window(f"{delimiter}5", [delimiter, digit])
    assert selected == words
    assert not trimmed


def test_trailing_sentence_punctuation_keeps_its_existing_timing():
    words = [Word(text="hello", start=1.0, end=1.2), Word(text="!", start=1.5, end=1.6)]
    cue = Cue(index=1, start_ms=900, end_ms=1800, lines=["hello!"])
    alignment = AlignmentResult(cue_word_indices={1: [0, 1]})
    rebuilt, flags = rebuild_cues([cue], words, alignment, StyleProfile(fps=30, min_cue_dur=.1))
    assert rebuilt[0].end_ms >= 1600
    assert cue_spoken_spans([cue], words, alignment)[1] == (1000, 1600)
    assert alignment.cue_word_indices == {1: [0, 1]}
    assert not any(flag.kind == "timing_outlier_trimmed" for flag in flags)


@pytest.mark.parametrize("neighbor_in_fragment", [False, True])
def test_leading_delimiter_keeps_a_source_fragment_with_no_own_lexical_word(neighbor_in_fragment):
    cue = Cue(index=1, start_ms=900, end_ms=1800, lines=["あの"])
    words = [
        Word(text="。", start=1.0, end=1.15), Word(text="の", start=1.4, end=1.6),
        Word(text="隣", start=1.7, end=1.8),
    ]
    alignment = AlignmentResult(
        cue_word_indices={1: [0, 1], 2: [2]},
        divergence_spans=[DivergenceSpan(
            case_id="fragment", cue_ids=[1, 2] if neighbor_in_fragment else [1],
            srt_text="あ隣" if neighbor_in_fragment else "あ", asr_text="。隣" if neighbor_in_fragment else "。",
            srt_token_indices=[0, 2] if neighbor_in_fragment else [0],
            asr_word_indices=[0, 2] if neighbor_in_fragment else [0],
        )],
    )
    original_ownership = alignment.model_dump()
    rebuilt, _ = rebuild_cues([cue], words, alignment, StyleProfile(fps=30, min_cue_dur=.1))
    refined, _ = refine_cues_to_speech_activity(
        rebuilt, [SpeechRegion(start=1.0, end=1.15), SpeechRegion(start=1.4, end=1.6)],
        StyleProfile(fps=30, min_cue_dur=.1), words=words, alignment=alignment,
    )
    assert refined[0].start_ms == 1000
    assert cue_spoken_spans([cue], words, alignment)[1] == (1000, 1600)
    assert alignment.model_dump() == original_ownership


def test_punctuation_backed_native_laugh_keeps_its_separate_final_burst():
    # 2B MAI case-53: native keep_srt/heard_clearly/confidence1/heard_text は
    # binds the third laugh to ASR558 。 and the burst at104.355–104.495.
    cue = Cue(index=75, start_ms=103834, end_ms=104567, lines=["ははは"])
    words = [
        Word(text="は", start=103.835, end=104.039), Word(text="は", start=104.120, end=104.265),
        Word(text="。", start=104.280, end=104.495),
    ]
    alignment = AlignmentResult(
        cue_word_indices={75: [0, 1, 2]}, divergence_spans=[DivergenceSpan(
            case_id="case-53", cue_ids=[75], srt_text="は", asr_text="。",
            srt_token_indices=[2], asr_word_indices=[2],
        )],
    )
    profile = StyleProfile(fps=30, min_cue_dur=.5)
    rebuilt, _ = rebuild_cues([cue], words, alignment, profile)
    refined, _ = refine_cues_to_speech_activity(
        rebuilt, [SpeechRegion(start=103.835, end=104.265), SpeechRegion(start=104.355, end=104.495)],
        profile, words=words, alignment=alignment,
    )
    assert refined[0].end_ms >= 104495
    assert cue_spoken_spans([cue], words, alignment)[75] == (103835, 104495)


@pytest.mark.parametrize("shared", [False, True])
def test_leading_punctuation_cannot_change_a_held_cues_overlap_evidence(shared):
    cues = [
        Cue(index=3, start_ms=0, end_ms=900, lines=["Previous"]),
        Cue(index=1, start_ms=1000, end_ms=1900, lines=["Hello"]),
        Cue(index=2, start_ms=1500, end_ms=3000, lines=["Later"]),
    ]
    words = [
        Word(text="Hello", start=1.0, end=1.8), Word(text=".", start=.5, end=.6),
        Word(text="Later", start=2.0, end=2.3), Word(text="Previous", start=.1, end=.6),
    ]
    alignment = AlignmentResult(cue_word_indices={1: [0], 2: [1, 2], 3: [1] if shared else [3]})
    options = {} if shared else {"protected_cue_ids": {2, 3}}
    spans = cue_spoken_spans(cues, words, alignment, **options)
    assert spans[2] == (500, 2300)
    final, flags = finalize_cues_for_output(
        cues, StyleProfile(fps=30), preserve_timing=True, protected_cue_ids={2, 3}, spoken_spans=spans,
    )
    assert next(cue for cue in final if cue.index == 2).start_ms == 1500
    assert any(flag.kind == "output_overlap_unresolved" for flag in flags)
