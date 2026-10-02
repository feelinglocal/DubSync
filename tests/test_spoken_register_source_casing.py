"""A spoken register form follows the script's case and punctuation (review F8).

The audio decides only which register form the actor used ("tá" for "está");
the customer's script decides sentence case and punctuation. Delivered ep17
#453 read "Tá tudo bem." after "Tudo já passou," and #722 read "tô ficando sem
dinheiro." after "Luke." because the raw ASR token was copied verbatim.
"""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.adjudication import AdjudicationEngine
from dubsync.adjudication_policy import DeterministicAdjudicationPolicy
from dubsync.aligner import align_cues_to_words
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import Cue, DivergenceSpan, Word
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile
from dubsync.tokenize import alphanumeric_signature


class NoCallAdapter:
    def adjudicate(self, spans):
        pytest.fail("A proven register equivalent must not call a provider")


class RecordingAdapter:
    def __init__(self):
        self.calls: list[list[str]] = []

    def adjudicate(self, spans):
        self.calls.append([span.case_id for span in spans])
        return [dict(case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text,
                     confidence=0.95, reason="Reviewed register wording.") for span in spans]


def _span(source: str, asr: str) -> DivergenceSpan:
    return DivergenceSpan(case_id="register", cue_ids=[1], srt_text=source, asr_text=asr,
                          start=1.0, end=2.0, speaker_ids=["speaker-1"])


@pytest.mark.parametrize(("language", "source", "asr", "expected"), [
    # Delivered ep17 source cues 448 and 709 (output #453 and #722).
    ("pt", "está", "Tá", "tá"),
    ("pt", "Estou", "tô", "Tô"),
    # Attached ASR punctuation is not script punctuation.
    ("pt", "para", "pra,", "pra"),
    ("pt", "vamos", '"Vamo', "vamo"),
    ("pt", "estava", "tava...", "tava"),
    ("pt", "Para", "pra", "Pra"),
    ("pt", "você", "Cê", "cê"),
    ("de", "habe", "Hab", "hab"),
    ("en", "going to", "Gonna,", "gonna"),
    # Multi-token spans keep every authored character outside the register words.
    ("pt", "Eu estou para sair.", "eu tô pra, sair!", "Eu tô pra sair."),
    ("pt", "para o", "Pro", "pro"),
    ("pt", "Pro", "para o", "Para o"),
    ("en", "I am going to leave.", "I am gonna leave!", "I am gonna leave."),
    ("en", "Gonna", "going to", "Going to"),
    ("de", "Ich habe es gesagt.", "ich hab es gesagt", "Ich hab es gesagt."),
    # All-caps source.
    ("pt", "ESTOU AQUI!", "tô aqui", "TÔ AQUI!"),
    ("pt", "PARA O", "pro", "PRO"),
    # Cue-initial words after an opening quotation mark or a dialogue dash.
    ("pt", '"Estou ficando sem dinheiro."', "tô ficando sem dinheiro", '"Tô ficando sem dinheiro."'),
    ("pt", "- Estou ficando sem dinheiro.", "tô ficando sem dinheiro.", "- Tô ficando sem dinheiro."),
    ("pt", "-está tudo bem?", "Tá tudo bem.", "-tá tudo bem?"),
])
def test_spoken_register_uses_source_case_and_punctuation(language, source, asr, expected):
    decisions, flags = AdjudicationEngine(NoCallAdapter(), language=language).adjudicate([_span(source, asr)])
    assert decisions[0].verdict == "use_audio"
    assert decisions[0].final_text == expected
    assert decisions[0].confidence == 1.0
    assert flags == []
    assert DeterministicAdjudicationPolicy(language=language).decide(_span(source, asr)).final_text == expected
    # Exactly the performed words, so word ownership and timing are unchanged.
    assert alphanumeric_signature(expected) == alphanumeric_signature(asr)


@pytest.mark.parametrize(("language", "source", "asr"), [
    # The ASR word carries more than the register word.
    ("de", "Ich hab's gesagt", "Ich habe's gesagt"),
    ("pt", "Eu vou para casa", "Eu vou pra,casa"),
    ("pt", "Eu estou bem", "Eu tô♪ bem"),
    # Authored punctuation inside the replaced words has no place in the register form.
    ("en", "I am going, to leave", "I am gonna leave"),
])
def test_register_word_with_other_content_is_left_to_adjudication(language, source, asr):
    adapter = RecordingAdapter()
    decisions, _ = AdjudicationEngine(adapter, language=language).adjudicate([_span(source, asr)])
    assert adapter.calls == [["register"]]
    assert decisions[0].final_text == source
    assert DeterministicAdjudicationPolicy(language=language).decide(_span(source, asr)) is None


def _applied(previous: str, target: str, spoken: list[str]):
    """Align, adjudicate and apply exactly as the pipeline does for one scene."""
    cues = [Cue(index=1, start_ms=0, end_ms=1900, lines=[previous]),
            Cue(index=2, start_ms=2000, end_ms=5000, lines=target.split("\n"))]
    previous_words = previous.split()
    words = [Word(text=text, start=index * 0.3, end=index * 0.3 + 0.25, speaker_id="speaker_0")
             for index, text in enumerate(previous_words)]
    words += [Word(text=text, start=2.05 + index * 0.3, end=2.3 + index * 0.3, speaker_id="speaker_0")
              for index, text in enumerate(spoken)]
    alignment = align_cues_to_words(cues, words, language="pt")
    spans = alignment.divergence_spans
    decisions, engine_flags = AdjudicationEngine(
        NoCallAdapter(), language="pt", source_cues=cues,
    ).adjudicate(spans)
    assert engine_flags == []
    assert {decision.verdict for decision in decisions} == {"use_audio"}
    changed, flags = apply_adjudication_decisions(
        cues, spans, decisions, StyleProfile(), words=words, token_matches=alignment.token_matches,
    )
    return {cue.index: cue for cue in changed}, flags


@pytest.mark.parametrize(("previous", "target", "spoken", "expected"), [
    # Delivered ep17 #452/#453 and #721/#722 (source cues 447/448 and 708/709).
    ("Tudo já passou,", "está tudo bem.", ["Tá", "tudo", "bem."], "tá tudo bem."),
    ("Luke.", "Estou ficando sem dinheiro.", ["tô", "ficando", "sem", "dinheiro."], "Tô ficando sem dinheiro."),
    # ep02-pt Scribe cue 731 and the synthetic stray comma.
    ("A gente tem que brindar, né?", "Para celebrar a volta do nosso senhorio.",
     ["pra", "celebrar", "a", "volta", "do", "nosso", "senhorio."], "Pra celebrar a volta do nosso senhorio."),
    ("Vamos.", "Eu vou para casa agora.", ["Eu", "vou", "pra,", "casa", "agora."], "Eu vou pra casa agora."),
    ("Vamos.", "Você está cansado.", ["Você", "Tá", "cansado."], "Você tá cansado."),
    ("Vamos.", "Eu estou indo para o mercado.", ["Eu", "Tô", "indo", "Pro", "mercado."], "Eu tô indo pro mercado."),
    ("Luke.", '"Estou ficando sem dinheiro."', ["tô", "ficando", "sem", "dinheiro."], '"Tô ficando sem dinheiro."'),
    ("Luke.", "- Estou ficando sem dinheiro.", ["tô", "ficando", "sem", "dinheiro."], "- Tô ficando sem dinheiro."),
    ("Luke.", "<i>Estou ficando sem dinheiro.</i>", ["tô", "ficando", "sem", "dinheiro."],
     "<i>Tô ficando sem dinheiro.</i>"),
    ("LUKE.", "ESTOU FICANDO SEM DINHEIRO.", ["tô", "ficando", "sem", "dinheiro."], "TÔ FICANDO SEM DINHEIRO."),
])
def test_applied_register_edit_keeps_the_cue_case_and_punctuation(previous, target, spoken, expected):
    cues, flags = _applied(previous, target, spoken)
    assert cues[1].plain_text == previous
    assert cues[2].plain_text == expected
    assert {flag.kind for flag in flags} == {"text_changed"}
    assert flags[-1].new_text == expected


def test_quoted_asr_register_word_is_not_an_editorial_guard_error():
    # ep02-pt Scribe saved cue 177: '"Vamo' was rejected with a severity-error item.
    cues, flags = _applied(
        "Conversei com seu pai,", "falei: vamos mandar um dinheiro para nossa filha.",
        ["falei:", '"Vamo', "mandar", "um", "dinheiro", "pra", "nossa", "filha."],
    )
    assert cues[2].plain_text == "falei: vamo mandar um dinheiro pra nossa filha."
    assert {flag.kind for flag in flags} == {"text_changed"}


class RecordingPipelineAdapter:
    def __init__(self):
        self.seen: list[DivergenceSpan] = []

    def adjudicate(self, spans):
        self.seen.extend(spans)
        return [dict(case_id=span.case_id, verdict="keep_srt", final_text=span.srt_text,
                     confidence=1, reason="Kept source.") for span in spans]


def test_default_pipeline_delivers_register_forms_in_source_case(tmp_path, monkeypatch):
    script = [
        ("Tudo já passou,", ["Tudo", "já", "passou,"]),
        ("está tudo bem.", ["Tá", "tudo", "bem."]),
        ("Luke.", ["Luke."]),
        ("Estou ficando sem dinheiro.", ["tô", "ficando", "sem", "dinheiro."]),
        ("Eu vou para casa agora.", ["Eu", "vou", "pra,", "casa", "agora."]),
        ("Para onde você vai?", ["pra", "onde", "você", "vai?"]),
        ("Ele disse vamos embora.", ["Ele", "disse", '"Vamo', "embora."]),
    ]
    cues = [Cue(index=index, start_ms=index * 3000, end_ms=index * 3000 + 2400, lines=[text])
            for index, (text, _) in enumerate(script, start=1)]
    words = [dict(text=token, start=cue.start_ms / 1000 + position * 0.4,
                  end=cue.start_ms / 1000 + position * 0.4 + 0.3, speaker_id="speaker_0")
             for cue, (_, tokens) in zip(cues, script) for position, token in enumerate(tokens)]
    source = tmp_path / "episode.srt"
    source.write_text(write_srt(cues), encoding="utf-8")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    config = tmp_path / "providers.yaml"
    # adjudication.register_policy is omitted on purpose: the default is spoken.
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}, "llm": {"provider": "fixture"}}),
                      encoding="utf-8")
    adapter = RecordingPipelineAdapter()
    monkeypatch.setattr(pipeline, "llm_adapter_from_config", lambda *a, **k: adapter)
    monkeypatch.setattr(pipeline, "punctuation_adapter_from_config", lambda *a, **k: None)
    output = tmp_path / "out.srt"
    result = pipeline.sync_episode(source, audio, output, tmp_path / "work", providers_path=config, language="pt")

    delivered = [cue.plain_text for cue in parse_srt_text(output.read_text(encoding="utf-8"))]
    assert delivered == [
        "Tudo já passou,", "tá tudo bem.", "Luke.", "Tô ficando sem dinheiro.",
        "Eu vou pra casa agora.", "Pra onde você vai?", "Ele disse vamo embora.",
    ]
    assert adapter.seen == []
    assert not [flag for flag in result.report["flags"] if flag["kind"] == "editorial_guard_rejected"]
