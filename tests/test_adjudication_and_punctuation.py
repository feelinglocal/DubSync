from __future__ import annotations

from dubsync.adjudication import AdjudicationEngine, StaticLLMAdapter
from dubsync.models import AudioSnippet, DivergenceSpan
from dubsync.punctuation import PunctuationValidationError, validate_punctuation_only


class SequencedLLMAdapter:
    def __init__(self, responses: list[list[dict[str, object]]]):
        self.responses = responses
        self.calls = 0

    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        self.calls += 1
        response_index = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[response_index]


class CountingLLMAdapter:
    def __init__(self):
        self.calls = 0

    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        self.calls += 1
        return []


class RecordingLLMAdapter:
    def __init__(self):
        self.batches: list[list[str]] = []

    def adjudicate(self, spans: list[DivergenceSpan]) -> list[dict[str, object]]:
        self.batches.append([span.case_id for span in spans])
        return [
            {
                "case_id": span.case_id,
                "verdict": "use_audio",
                "final_text": span.asr_text,
                "confidence": 0.9,
                "speaker": None,
                "character": "unknown",
                "reason": "spoken words differ from source",
            }
            for span in spans
        ]


class RecordingSnippetAwareAdapter:
    def __init__(self):
        self.snippets_by_case: dict[str, AudioSnippet] = {}

    def adjudicate_with_audio(
        self,
        spans: list[DivergenceSpan],
        audio_snippets: dict[str, AudioSnippet],
    ) -> list[dict[str, object]]:
        self.snippets_by_case = dict(audio_snippets)
        return [
            {
                "case_id": span.case_id,
                "verdict": "use_audio",
                "final_text": span.asr_text,
                "confidence": 0.91,
                "speaker": None,
                "character": "unknown",
                "reason": "audio snippet confirms the spoken line",
            }
            for span in spans
        ]


def make_span(case_id: str, start: float, end: float) -> DivergenceSpan:
    return DivergenceSpan(
        case_id=case_id,
        cue_ids=[int(case_id.rsplit("-", 1)[-1])],
        srt_text=f"source line {case_id}",
        asr_text=f"spoken rewrite {case_id}",
        start=start,
        end=end,
        confidence=0.82,
        speaker_ids=[],
    )


def test_llm_adjudication_batches_spans_by_scene_gap():
    llm = RecordingLLMAdapter()
    spans = [
        make_span("case-1", 0.0, 1.0),
        make_span("case-2", 2.0, 3.0),
        make_span("case-3", 8.0, 9.0),
    ]

    decisions, flags = AdjudicationEngine(llm, scene_gap_seconds=4.0).adjudicate(spans)

    assert llm.batches == [["case-1", "case-2"], ["case-3"]]
    assert [decision.case_id for decision in decisions] == ["case-1", "case-2", "case-3"]
    assert flags == []


def test_adjudication_passes_audio_snippets_to_snippet_aware_adapter(tmp_path):
    adapter = RecordingSnippetAwareAdapter()
    span = make_span("case-1", 1.0, 2.0)
    snippet = AudioSnippet(
        case_id="case-1",
        path=str(tmp_path / "case-1.wav"),
        mime_type="audio/wav",
        start=0.0,
        end=4.0,
    )

    decisions, flags = AdjudicationEngine(adapter, audio_snippets={"case-1": snippet}).adjudicate([span])

    assert adapter.snippets_by_case == {"case-1": snippet}
    assert decisions[0].reason == "audio snippet confirms the spoken line"
    assert flags == []


def test_adjudication_uses_audio_text_and_flags_low_confidence():
    span = DivergenceSpan(
        case_id="case-1",
        cue_ids=[3],
        srt_text="old line",
        asr_text="new line",
        start=1.0,
        end=2.0,
        confidence=0.92,
        speaker_ids=["speaker_1"],
    )
    llm = StaticLLMAdapter(
        {
            "case-1": {
                "case_id": "case-1",
                "verdict": "use_audio",
                "final_text": "new line",
                "confidence": 0.61,
                "speaker": "speaker_1",
                "character": "unknown",
                "reason": "spoken words differ from the source SRT",
            }
        }
    )

    decisions, flags = AdjudicationEngine(llm, confidence_gate=0.7).adjudicate([span])

    assert decisions[0].verdict == "use_audio"
    assert decisions[0].final_text == "new line"
    assert flags[0].kind == "low_confidence_adjudication"


def test_punctuation_only_diff_keeps_srt_without_llm_call():
    span = DivergenceSpan(
        case_id="case-punct",
        cue_ids=[7],
        srt_text="Hello, there!",
        asr_text="hello there",
        start=1.0,
        end=2.0,
        confidence=0.94,
        speaker_ids=["speaker_1"],
    )
    llm = CountingLLMAdapter()

    decisions, flags = AdjudicationEngine(llm).adjudicate([span])

    assert llm.calls == 0
    assert decisions[0].verdict == "keep_srt"
    assert decisions[0].final_text == "Hello, there!"
    assert decisions[0].reason == "Punctuation/casing-only difference; preserved source SRT."
    assert flags == []


def test_tiny_asr_noise_keeps_srt_without_llm_call():
    span = DivergenceSpan(
        case_id="case-noise",
        cue_ids=[8],
        srt_text="general kenobi",
        asr_text="general kenoby",
        start=2.0,
        end=3.0,
        confidence=0.91,
        speaker_ids=[],
    )
    llm = CountingLLMAdapter()

    decisions, flags = AdjudicationEngine(llm).adjudicate([span])

    assert llm.calls == 0
    assert decisions[0].verdict == "keep_srt"
    assert decisions[0].final_text == "general kenobi"
    assert decisions[0].reason == "Tiny ASR spelling/noise difference; preserved source SRT."
    assert flags == []


def test_invalid_llm_payload_retries_once_and_uses_valid_retry():
    span = DivergenceSpan(
        case_id="case-2",
        cue_ids=[4],
        srt_text="source text",
        asr_text="spoken text",
        start=1.0,
        end=2.0,
        confidence=0.80,
        speaker_ids=[],
    )
    llm = SequencedLLMAdapter(
        [
            [{"case_id": "case-2", "verdict": "bad"}],
            [
                {
                    "case_id": "case-2",
                    "verdict": "use_audio",
                    "final_text": "spoken text",
                    "confidence": 0.91,
                    "speaker": None,
                    "character": "unknown",
                    "reason": "retry returned valid structured output",
                }
            ],
        ]
    )

    decisions, flags = AdjudicationEngine(llm).adjudicate([span])

    assert llm.calls == 2
    assert decisions[0].verdict == "use_audio"
    assert decisions[0].final_text == "spoken text"
    assert "invalid_llm_response" not in {flag.kind for flag in flags}


def test_invalid_llm_payload_degrades_after_retry_with_qc_flag():
    span = DivergenceSpan(
        case_id="case-2",
        cue_ids=[4],
        srt_text="source text",
        asr_text="spoken text",
        start=1.0,
        end=2.0,
        confidence=0.80,
        speaker_ids=[],
    )
    llm = SequencedLLMAdapter(
        [
            [{"case_id": "case-2", "verdict": "bad"}],
            [{"case_id": "case-2", "verdict": "still_bad"}],
        ]
    )

    decisions, flags = AdjudicationEngine(llm).adjudicate([span])

    assert llm.calls == 2
    assert decisions[0].verdict == "keep_srt"
    assert decisions[0].final_text == "source text"
    assert "invalid_llm_response" in {flag.kind for flag in flags}


def test_punctuation_validator_accepts_case_and_punctuation_changes_only():
    assert validate_punctuation_only("hello there", "Hello, there.") == "Hello, there."


def test_punctuation_validator_rejects_word_changes():
    try:
        validate_punctuation_only("hello there", "Hello, world.")
    except PunctuationValidationError as exc:
        assert "alphanumeric content changed" in str(exc)
    else:
        raise AssertionError("word-changing punctuation pass should fail")


def test_punctuation_validator_rejects_digit_word_substitution():
    try:
        validate_punctuation_only("you have 2 choices", "You have two choices.")
    except PunctuationValidationError as exc:
        assert "alphanumeric content changed" in str(exc)
    else:
        raise AssertionError("digit-to-word substitution should fail word-freeze validation")
