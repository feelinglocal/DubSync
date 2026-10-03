import json
import sys
import types
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.adjudication import AdjudicationEngine, confidence_gated_decision
from dubsync.llm_providers import OpenAILLMAdapter
from dubsync.models import AdjudicationDecision, AudioSnippet, Cue, DivergenceSpan
from dubsync.qc_review import build_review
from dubsync.srt_io import parse_srt_text


@pytest.mark.parametrize("evidence,heard", [("heard_unclear", "maybe"), ("not_audible", "")])
def test_uncertain_audio_is_held_even_when_numeric_gate_is_disabled(evidence, heard):
    span = DivergenceSpan(case_id="case", cue_ids=[1], srt_text="original", asr_text="maybe")
    decision = AdjudicationDecision(case_id="case", verdict="use_audio", final_text=heard,
        confidence=1, reason="uncertain", evidence=evidence, heard_text=heard)
    selected, flag = confidence_gated_decision(span, decision, 0)
    assert selected.verdict == "keep_srt" and selected.final_text == "original"
    assert flag is not None and flag.kind == "low_confidence_adjudication"


_SOURCE = [Cue(index=2, start_ms=26_500, end_ms=27_365, lines=["いいかしら？"]),
           Cue(index=3, start_ms=32_300, end_ms=32_630, lines=["あなたは…"])]


def _insertion_answer(source, asr, evidence, heard, verdict="keep_srt"):
    # Delivered 1B-mai case-47: '。' between two cues, answered "no spoken dialogue exists".
    span = DivergenceSpan(case_id="case-47", cue_ids=[3] if source else [], srt_text=source, asr_text=asr,
                          start=27.44, end=27.52)
    decision = AdjudicationDecision(
        case_id=span.case_id, verdict=verdict, final_text=source if verdict == "keep_srt" else "",
        confidence=0, reason="ASR proposes punctuation where no spoken dialogue exists.",
        evidence=evidence, heard_text=heard,
    )
    return span, decision


@pytest.mark.parametrize("verdict", ["keep_srt", "use_audio"])
@pytest.mark.parametrize("evidence", ["not_audible", "heard_unclear"])
@pytest.mark.parametrize("asr", ["。", "、", "?"])
def test_agreed_absence_on_a_punctuation_only_insertion_is_not_a_hold(asr, evidence, verdict):
    span, decision = _insertion_answer("", asr, evidence, "", verdict)

    selected, flag = confidence_gated_decision(span, decision, .7)

    assert flag is None
    assert (selected.verdict, selected.final_text) == ("keep_srt", "")


@pytest.mark.parametrize("source,asr,evidence,heard", [
    ("あなたは", "。", "not_audible", ""),  # source words are not confirmed absent
    ("", "Oh.", "not_audible", ""),  # the ASR heard a word the model could not recover
    ("", "う", "heard_unclear", ""),
    ("", "。", "heard_unclear", "あ"),  # the model heard something
])
def test_uncertain_hearing_with_a_spoken_side_stays_customer_review(source, asr, evidence, heard):
    span, decision = _insertion_answer(source, asr, evidence, heard)

    selected, flag = confidence_gated_decision(span, decision, .7)

    assert (selected.verdict, selected.final_text) == ("keep_srt", source)
    assert flag is not None and flag.kind == "low_confidence_adjudication"
    review = build_review([flag], [], _SOURCE, source_cues=_SOURCE)
    assert [item.raw_flags for item in review.review] == [[0]]


_2B_CUE = [Cue(index=79, start_ms=109_100, end_ms=110_900, lines=["大井周治 いい度胸だな！"])]


def _2b_scribe_case_36(final_text, *, srt_text="大井周治", asr_text="おい秀二"):
    # Delivered 2B-scribe case-36: the MAI run of the same audio hears 大井周治 clearly.
    span = DivergenceSpan(case_id="case-36", cue_ids=[79], srt_text=srt_text, asr_text=asr_text,
                          start=109.16, end=109.9, srt_token_indices=[547, 548, 549, 550],
                          asr_word_indices=[582, 583, 585, 586])
    decision = AdjudicationDecision(
        case_id="case-36", verdict="hybrid", final_text=final_text, confidence=1.0,
        reason="Speaker says interjection 'おい' followed by the character's name '周治'.",
        evidence="heard_clearly", heard_text=final_text,
    )
    return span, decision


def test_a_clear_hearing_that_respells_source_kanji_in_kana_is_held_for_review():
    # W3R-3 / F10: 大井 and おい sound alike; the hearing cannot tell the customer's name from the
    # interjection, so the confident rewrite 大井周治 -> おい周治 is held, not applied silently.
    span, decision = _2b_scribe_case_36("おい周治")

    selected, flag = confidence_gated_decision(span, decision, .7)

    assert (selected.verdict, selected.final_text) == ("keep_srt", "大井周治")
    assert flag is not None and flag.kind == "low_confidence_adjudication"
    assert (flag.old_text, flag.new_text, flag.cue_ids) == ("大井周治", "おい周治", [79])
    review = build_review([flag], [], _2B_CUE, source_cues=_2B_CUE)
    assert [item.raw_flags for item in review.review] == [[0]]


@pytest.mark.parametrize("srt_text,final_text", [
    ("山下様", "山下さん"),  # one kanji: a real honorific change (delivered 1B 様 -> さん)
    ("いい度胸", "いい根性"),  # kanji to kanji stays a model decision
    ("そうだね", "そうだよ"),  # kana only
    ("大井周治", "大井周治さん"),  # an addition keeps every kanji
])
def test_other_confident_rewrites_still_apply(srt_text, final_text):
    span, decision = _2b_scribe_case_36(final_text, srt_text=srt_text, asr_text=final_text)

    selected, flag = confidence_gated_decision(span, decision, .7)

    assert flag is None
    assert selected.final_text == final_text


def _text_only_openai(monkeypatch, prompts):
    """The real OpenAI adapter over a fake SDK whose model claims to hear the ASR wording."""
    class Response:
        status = "completed"

        def __init__(self, parsed):
            self.output_parsed, self.usage, self.output = parsed, {"input_tokens": 10, "output_tokens": 5}, []

    class Responses:
        def parse(self, **kwargs):
            payload = json.loads(kwargs["input"])
            prompts.append(payload)
            return Response(kwargs["text_format"].model_validate({"decisions": [dict(
                case_id=case["case_id"], verdict="use_audio", final_text=case["asr_text"],
                heard_text=case["asr_text"], evidence="heard_clearly", speaker=None, character="unknown",
                reason="Clearly heard.",
            ) for case in payload["spans"]]}))

    class Client:
        def __init__(self, **_kwargs):
            self.responses = Responses()

    module = types.ModuleType("openai")
    module.OpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", module)


def _orange_span(case_id="case-1", cue_id=4, start=8.0):
    return DivergenceSpan(case_id=case_id, cue_ids=[cue_id], srt_text="old orange anchor",
                          asr_text="fresh orange anchor", start=start, end=start + 1)


def test_a_text_only_route_holds_a_claimed_hearing_for_review(monkeypatch):
    # Fable review F22: no audio was attached, yet heard_clearly became confidence 1.0.
    prompts = []
    _text_only_openai(monkeypatch, prompts)
    span = _orange_span()

    decisions, flags = AdjudicationEngine(
        OpenAILLMAdapter(api_key="test-key"), require_audio_for_hearing=True).adjudicate([span])

    assert [payload["audio_snippets"] for payload in prompts] == [[]]
    held = decisions[0]
    assert (held.verdict, held.final_text, held.confidence, held.evidence, held.heard_text) == (
        "keep_srt", "old orange anchor", 0.0, None, None)
    assert [(flag.kind, flag.cue_ids, flag.old_text, flag.new_text) for flag in flags] == [
        ("adjudication_hearing_unverified", [4], "old orange anchor", "fresh orange anchor")]
    review = build_review(flags, [], [Cue(index=4, start_ms=8000, end_ms=9000, lines=["old orange anchor"])])
    assert [item.raw_flags for item in review.review] == [[0]]


def test_only_the_case_without_a_covering_clip_is_held(tmp_path):
    clipped, unclipped = _orange_span(), _orange_span("case-2", 6, 12.0)
    clip = tmp_path / "case-1.wav"
    clip.write_bytes(b"clip")

    class Adapter:
        def adjudicate_with_audio(self, spans, snippets):
            return [dict(case_id=span.case_id, verdict="use_audio", final_text=span.asr_text,
                         heard_text=span.asr_text, evidence="heard_clearly", reason="heard") for span in spans]

    engine = AdjudicationEngine(
        Adapter(), require_audio_for_hearing=True,
        audio_snippets={"case-1": AudioSnippet(case_id="case-1", path=str(clip), start=7.5, end=9.5)})
    decisions, flags = engine.adjudicate([clipped, unclipped])

    assert [(item.verdict, item.final_text) for item in decisions] == [
        ("use_audio", "fresh orange anchor"), ("keep_srt", "old orange anchor")]
    assert [(flag.kind, flag.cue_ids) for flag in flags] == [("adjudication_hearing_unverified", [6])]


def test_text_only_openai_route_never_applies_unheard_wording(tmp_path, monkeypatch):
    # The documented text default (llm.provider: openai) sends no clip and no episode audio.
    prompts = []
    _text_only_openai(monkeypatch, prompts)
    lines = ["old orange anchor", "middle cue number two", "old purple anchor"]
    source = tmp_path / "episode.srt"
    source.write_text("".join(f"{index}\n00:00:0{index * 2},000 --> 00:00:0{index * 2 + 1},500\n{line}\n\n"
                              for index, line in enumerate(lines, 1)), encoding="utf-8")
    words = [dict(text=token, start=index * 2 + position * .3, end=index * 2 + position * .3 + .25)
             for index, line in enumerate(lines, 1)
             for position, token in enumerate(line.replace("old", "fresh").split())]
    (tmp_path / "words.json").write_text(json.dumps({"words": words}), encoding="utf-8")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((1000).to_bytes(2, "little", signed=True) * 16000 * 9)
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({
        "asr": {"fixture_path": str(tmp_path / "words.json")},
        "llm": {"provider": "openai", "api_key": "test-key", "model": "gpt-5.6-luna"},
    }), encoding="utf-8")
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "speaker_mapping_adapter_from_config", lambda *_args, **_kwargs: None)

    result = pipeline.sync_episode(source, audio, tmp_path / "out.srt", tmp_path / "work",
                                   providers_path=providers, language="en")

    assert prompts and all(payload["audio_snippets"] == [] for payload in prompts)
    assert [cue.plain_text for cue in parse_srt_text((tmp_path / "out.srt").read_text(encoding="utf-8"))] == lines
    held = [flag for flag in result.report["flags"] if flag["kind"] == "adjudication_hearing_unverified"]
    assert sorted(cue for flag in held for cue in flag["cue_ids"]) == [1, 3]
    decisions = json.loads((result.episode_workdir / "adjudicate.json").read_text(encoding="utf-8"))["decisions"]
    assert all(item.get("evidence") is None for item in decisions)
