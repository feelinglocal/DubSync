from __future__ import annotations

import pytest

from dubsync.adjudication import AdjudicationEngine
from dubsync.models import Cue, DivergenceSpan


class RecordingAdapter:
    def __init__(self):
        self.calls: list[list[str]] = []

    def adjudicate(self, spans):
        self.calls.append([span.case_id for span in spans])
        return [
            {
                "case_id": span.case_id,
                "verdict": "keep_srt",
                "final_text": span.srt_text,
                "confidence": 0.95,
                "reason": "Reviewed ambiguous wording.",
            }
            for span in spans
        ]


def span(source: str, audio: str) -> DivergenceSpan:
    return DivergenceSpan(
        case_id="case-policy", cue_ids=[1], srt_text=source, asr_text=audio,
        start=1.0, end=2.0, confidence=0.0, speaker_ids=["speaker-1"],
        asr_word_indices=[3, 4], srt_token_indices=[1, 2],
    )


def cues(*texts: str) -> list[Cue]:
    return [
        Cue(index=index, start_ms=index * 1000, end_ms=index * 1000 + 900, lines=text.split("\n"))
        for index, text in enumerate(texts, start=1)
    ]


def decide(source: str, audio: str, **kwargs):
    adapter = RecordingAdapter()
    original = span(source, audio)
    before = original.model_dump()
    decisions, flags = AdjudicationEngine(adapter, **kwargs).adjudicate([original])
    assert original.model_dump() == before
    return decisions[0], flags, adapter


@pytest.mark.parametrize(
    ("source", "audio", "language"),
    [
        ("Feliz Ano-Novo!", "Feliz ano novo", "pt"),
        ("bem-sucedida", "bem sucedida", "pt-BR"),
        ("superpreocupado", "super preocupado", "pt"),
        ("E-Mail", "email", "de"),
        ("Ich hab's gesagt", "Ich habs gesagt", "de"),
        ("So wär's besser", "So wärs besser", "de"),
        ("e-mail", "email", "en"),
    ],
)
def test_proven_spacing_forms_preserve_source_without_llm(source, audio, language):
    decision, flags, adapter = decide(source, audio, language=language)
    assert adapter.calls == []
    assert decision.verdict == "keep_srt"
    assert decision.final_text == source
    assert decision.confidence == 1.0
    assert flags == []


@pytest.mark.parametrize(
    ("source", "audio", "language"),
    [
        ("Sr.", "senhor", "pt"),
        ("Srta. Shang", "senhorita Shang", "pt"),
        ("Dra. Silva", "doutora Silva", "pt"),
        ("Hr. Schmidt", "Herr Schmidt", "de"),
        ("Fr. Schmidt", "Frau Schmidt", "de-DE"),
        ("Mr. Smith", "Mister Smith", "en"),
        ("Dr. Smith", "Doctor Smith", "en"),
    ],
)
def test_language_scoped_abbreviation_keeps_exact_source(source, audio, language):
    decision, flags, adapter = decide(source, audio, language=language)
    assert adapter.calls == []
    assert decision.final_text == source
    assert decision.confidence == 1.0
    assert flags == []


@pytest.mark.parametrize(
    ("source", "audio", "language"),
    [
        ("para", "pra", "pt"),
        ("para o", "pro", "pt"),
        ("está", "tá", "pt"),
        ("estou", "tô", "pt"),
        ("estava", "tava", "pt"),
        ("você", "cê", "pt"),
        ("Ich habe es", "Ich hab es", "de"),
        ("I am going to leave", "I am gonna leave", "en"),
    ],
)
def test_explicit_script_register_keeps_source(source, audio, language):
    decision, flags, adapter = decide(source, audio, language=language, register_policy="script")
    assert adapter.calls == []
    assert decision.verdict == "keep_srt"
    assert decision.final_text == source
    assert decision.confidence == 1.0
    assert flags == []


def test_script_keep_preserves_markup_linebreaks_and_has_no_audio_dependency():
    source = "<i>Eu estou\npara sair.</i>"
    decision, flags, adapter = decide(
        source, "Eu tô pra sair", language="pt", confidence_gate=1.0,
        require_audio_snippets=True, register_policy="script",
    )
    assert adapter.calls == []
    assert decision.final_text == source
    assert decision.speaker == "speaker-1"
    assert decision.confidence == 1.0
    assert not {"start", "end"}.intersection(decision.model_dump())
    assert flags == []


@pytest.mark.parametrize(("source", "audio", "expected"), [
    # The performed register words, with the script's punctuation (review F8).
    ("Eu estou para sair", "Eu tô pra sair!", "Eu tô pra sair"),
    ("Eu tô aqui", "Eu estou aqui.", "Eu estou aqui"),
])
def test_spoken_register_uses_only_exact_proven_asr_text(source, audio, expected):
    decision, flags, adapter = decide(source, audio, language="pt", register_policy="spoken")
    assert adapter.calls == []
    assert decision.verdict == "use_audio"
    assert decision.final_text == expected
    assert decision.confidence == 1.0
    assert flags == []


@pytest.mark.parametrize(
    ("source", "audio"),
    [
        ("Eu estou aqui", "Eu tô lá"),
        ("Eu estou aqui", "Eu não tô aqui"),
        ("Sr. está aqui", "Senhor tá aqui"),
        ("<i>Eu estou aqui</i>", "Eu tô aqui"),
        ("Eu estou\naqui", "Eu tô aqui"),
    ],
)
def test_spoken_policy_leaves_mixed_or_layout_changes_to_review(source, audio):
    _, _, adapter = decide(source, audio, language="pt", register_policy="spoken")
    assert adapter.calls == [["case-policy"]]


@pytest.mark.parametrize("language", [None, "auto", "ja", "de"])
def test_portuguese_rules_are_never_inferred_from_unknown_or_other_language(language):
    for source, audio in [("para", "pra"), ("Sr", "senhor"), ("Ano-Novo", "ano novo")]:
        _, _, adapter = decide(source, audio, language=language)
        assert adapter.calls == [["case-policy"]]


@pytest.mark.parametrize(
    ("source", "audio", "language"),
    [
        ("re-sign", "resign", "en"),
        ("re-cover", "recover", "en"),
        ("un-ionized", "unionized", "en"),
        ("co-op", "coop", "en"),
        ("a part", "apart", "en"),
        ("now here", "nowhere", None),
        ("por que", "porque", "pt"),
        ("1 23", "12 3", "pt"),
        ("1-2", "12", "pt"),
        ("Sr. 11", "senhor 12", "pt"),
        ("não está", "tá", "pt"),
        ("nicht", "ist", "de"),
    ],
)
def test_semantic_compounds_numbers_and_negations_still_reach_review(source, audio, language):
    _, _, adapter = decide(source, audio, language=language)
    assert adapter.calls == [["case-policy"]]


@pytest.mark.parametrize(
    ("source", "audio", "source_cues"),
    [
        ("Dony", "Donnie", cues("O Dony chegou.", "Chame o Dony.")),
        ("Luan Nian", "Luanian", cues("Fale com Luan Nian.", "Eu vi Luan Nian.")),
        ("Hang Zhou", "Hangzhou", cues("Eu moro em Hang Zhou.", "Voltei de Hang Zhou.")),
    ],
)
def test_recurring_mid_sentence_source_names_keep_script(source, audio, source_cues):
    before = [cue.model_dump() for cue in source_cues]
    decision, flags, adapter = decide(source, audio, language="pt", source_cues=source_cues)
    assert adapter.calls == []
    assert decision.final_text == source
    assert decision.confidence == 1.0
    assert flags == []
    assert [cue.model_dump() for cue in source_cues] == before


@pytest.mark.parametrize(
    ("source", "audio", "source_cues", "language"),
    [
        ("Kenobi", "Kenoby", cues("General Kenobi chegou."), "pt"),
        ("Kenobi", "Kenoby", cues("Kenobi chegou.", "Kenobi saiu."), "pt"),
        ("Kenobi", "Kenoby", cues("Eu vi Kenobi e Kenobi."), "pt"),
        ("Dony", "Donnie", cues("O Dony chegou.", "Vi Dony e Donnie.", "Chame Donnie."), "pt"),
        ("Dony", "Dani", cues("O Dony chegou.", "Chame o Dony."), "pt"),
        ("Três", "Trêss", cues("Chame Três.", "Vi Três."), "pt"),
        ("Dony", "Donnie", cues("Eu vi Dony.", "Chame Dony.", "A palavra dony é comum."), "pt"),
        ("Dony", "Donnie", cues("Ele foi. Dony veio.", "Ele saiu. Dony entrou."), "pt"),
        ("Dony está", "Donnie estava", cues("O Dony chegou.", "Chame o Dony."), "pt"),
        ("Arbeit", "Arbeitx", cues("Die Arbeit beginnt.", "Meine Arbeit endet."), "de"),
        ("Weise", "Weize", cues("Die Weise singt.", "Eine Weise singt."), "de"),
        ("Dony", "Donnie", cues("O Dony chegou.", "Chame o Dony."), None),
    ],
)
def test_unproven_or_ambiguous_names_do_not_override_review(source, audio, source_cues, language):
    _, _, adapter = decide(source, audio, source_cues=source_cues, language=language)
    assert adapter.calls == [["case-policy"]]


def test_german_name_requires_repeated_title_evidence():
    source_cues = cues("Bitte Herrn Kenobi fragen.", "Das weiß Herr Kenobi.")
    decision, flags, adapter = decide("Kenobi", "Kenoby", source_cues=source_cues, language="de")
    assert adapter.calls == []
    assert decision.final_text == "Kenobi"
    assert flags == []


@pytest.mark.parametrize("register_policy", ["audio", "", None, True])
def test_invalid_register_policy_is_rejected(register_policy):
    with pytest.raises(ValueError, match="register_policy"):
        AdjudicationEngine(RecordingAdapter(), register_policy=register_policy)


def test_original_punctuation_only_policy_remains_available_without_language():
    decision, flags, adapter = decide("Hello,\nthere!", "hello there")
    assert adapter.calls == []
    assert decision.final_text == "Hello,\nthere!"
    assert decision.reason == "Punctuation/casing-only difference; preserved source SRT."
    assert flags == []


@pytest.mark.parametrize(("source", "audio"), [("N-não", "não"), ("De-deixa", "deixa"), ("eu-eu", "eu")])
def test_existing_glued_stutter_equivalence_is_preserved(source, audio):
    decision, flags, adapter = decide(source, audio)
    assert adapter.calls == []
    assert decision.final_text == source
    assert flags == []


def test_name_lexicon_is_reusable_and_requires_known_language():
    from dubsync.adjudication_policy import build_source_name_lexicon

    source_cues = cues("O Dony chegou.", "Chame o Dony.")
    assert build_source_name_lexicon(source_cues, "pt-BR") == frozenset({("dony",)})
    assert build_source_name_lexicon(source_cues) == frozenset()
