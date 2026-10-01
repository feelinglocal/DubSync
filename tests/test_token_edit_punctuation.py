"""Token edits keep the punctuation that belongs to the text they do not touch."""
from __future__ import annotations

import json

import pytest
import yaml

from dubsync import pipeline
from dubsync.changes import apply_adjudication_decisions
from dubsync.models import AdjudicationDecision, Cue, DivergenceSpan
from dubsync.srt_io import parse_srt_text
from dubsync.style_profile import StyleProfile


def _edit(text: str, tokens: list[int], srt_text: str, final_text: str, *, asr_text: str | None = None):
    cue = Cue(index=1, start_ms=0, end_ms=3000, lines=[text])
    span = DivergenceSpan(
        case_id="case-1", cue_ids=[1], srt_text=srt_text, asr_text=final_text if asr_text is None else asr_text,
        srt_token_indices=tokens, asr_word_indices=[0] if final_text else [],
    )
    decision = AdjudicationDecision(case_id="case-1", verdict="use_audio", final_text=final_text,
                                    confidence=1.0, reason="heard")
    cues, flags = apply_adjudication_decisions([cue], [span], [decision], StyleProfile())
    return cues[0].plain_text, [flag.kind for flag in flags]


# --- apostrophes ------------------------------------------------------------------------------


@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_replacing_the_word_after_an_elision_keeps_the_apostrophe(apostrophe):
    # rebuild.md BUG-11: "Je vois l'homme ici." became "Je vois l enfant ici."
    text, kinds = _edit(f"Je vois l{apostrophe}homme ici.", [3], "homme", "enfant")

    assert text == f"Je vois l{apostrophe}enfant ici."
    assert kinds == ["text_changed"]


def test_replacing_a_single_quoted_word_keeps_both_quotes():
    # "Ele disse 'oi' para mim." became "Ele disse olá' para mim." and was then rejected.
    text, kinds = _edit("Ele disse 'oi' para mim.", [2], "oi", "olá")

    assert text == "Ele disse 'olá' para mim."
    assert kinds == ["text_changed"]


def test_expanding_a_contraction_suffix_still_drops_its_apostrophe():
    assert _edit("Hier gibt's keine Monster.", [2], "s", "es")[0] == "Hier gibt es keine Monster."
    assert _edit("I can't go there.", [2], "t", "not")[0] == "I can not go there."


# --- quotes -----------------------------------------------------------------------------------

_QUOTED = 'Olha para a frente, sua "cheirosa de pêssego".'


def test_deleting_a_phrase_that_starts_inside_a_quotation_leaves_no_orphan_quote():
    # Scribe ep11 cue 228: the remainder was 'Olha para a frente, ".', which the
    # cue-level guard rejected together with every other approved edit of the cue.
    text, kinds = _edit(_QUOTED, [4, 5, 6, 7], "sua cheirosa de pêssego", "", asr_text="")

    assert text == "Olha para a frente."
    assert kinds == ["text_changed"]


def test_replacing_a_word_inside_a_quotation_keeps_the_quotation():
    assert _edit(_QUOTED, [7], "pêssego", "maçã")[0] == 'Olha para a frente, sua "cheirosa de maçã".'


@pytest.mark.parametrize("final_text", ['maçã"', 'Olha para a frente, sua "cheirosa de maçã".'])
def test_quote_the_source_cue_already_has_is_not_an_editorial_addition(final_text):
    # rebuild.md BUG-16: the span text has no punctuation, so an echoed quote of
    # the source cue was rejected as a changed quotation signature.
    text, kinds = _edit(_QUOTED, [7], "pêssego", final_text)

    assert text == 'Olha para a frente, sua "cheirosa de maçã".'
    assert kinds == ["text_changed"]


@pytest.mark.parametrize("text, tokens, source, final_text", [
    ("Olha a cheirosa aqui.", [2], "cheirosa", '"linda"'),
    # A partly repeated quotation cannot be told from a second one.
    (_QUOTED, [7], "pêssego", '"cheirosa de maçã"'),
])
def test_quote_the_source_does_not_have_at_that_place_is_still_rejected(text, tokens, source, final_text):
    edited, kinds = _edit(text, tokens, source, final_text)

    assert edited == text
    assert kinds == ["editorial_guard_rejected"]


# --- abbreviations ----------------------------------------------------------------------------


def test_replacing_a_title_abbreviation_does_not_keep_its_period_inside_the_sentence():
    # golden.md 6.4: 'Bom dia, senhor. Luan.'
    assert _edit("Bom dia, Sr. Luan.", [2], "Sr", "seu")[0] == "Bom dia, seu Luan."
    assert _edit("Olá, estou procurando o Sr. Luan.", [4], "Sr", "chefe")[0] == "Olá, estou procurando o chefe Luan."


def test_title_abbreviation_at_the_end_of_the_sentence_keeps_the_sentence_period():
    assert _edit("Bom dia, Sr.", [2], "Sr", "seu")[0] == "Bom dia, seu."


@pytest.mark.parametrize("text, tokens, source, spoken", [
    ("Bom dia, Sr. Luan.", [2], "Sr", "senhor"),
    ("A Srta. Shang chegou.", [1], "Srta", "senhorita"),
    ("O Dr. Paulo já vem.", [1], "Dr", "doutor"),
])
def test_abbreviation_and_its_spoken_form_are_the_same_word(text, tokens, source, spoken):
    # The reviewer kept "Sr." / "Srta." in 9 of 9 places where the actor says the word.
    edited, kinds = _edit(text, tokens, source, spoken)

    assert edited == text
    assert kinds == []


# --- removed clauses --------------------------------------------------------------------------


@pytest.mark.parametrize("text, tokens, removed, expected", [
    ("Vou chamar a polícia, você me bateu.", [4, 5, 6], "você me bateu", "Vou chamar a polícia."),
    ("E você, estava com medo?", [2, 3, 4], "estava com medo", "E você?"),
    ("a oferecer, dentro do possível,", [2, 3, 4], "dentro do possível", "a oferecer,"),
    ("ligadas a ele, com valores inflados.", [3, 4, 5], "com valores inflados", "ligadas a ele."),
])
def test_removed_clause_leaves_no_dangling_or_doubled_punctuation(text, tokens, removed, expected):
    # corpus.md B3: '...ligados a ele,.', 'Eu Vou chamar a polícia,.', 'a oferecer,,'
    assert _edit(text, tokens, removed, "", asr_text="")[0] == expected


def test_authored_punctuation_clusters_are_not_rewritten():
    assert _edit("Bem,... a Corp Ltd., sabe, falhou.", [4], "sabe", "viu")[0] == "Bem,... a Corp Ltd., viu, falhou."


# --- leftover duplicate cue -------------------------------------------------------------------


def _sync(tmp_path, srt: str, words: list[tuple], *, no_llm: bool = True):
    source = tmp_path / "episode.srt"
    source.write_text(srt, encoding="utf-8")
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [
        {"text": text, "start": start, "end": end, "confidence": None} for text, start, end in words
    ]}, ensure_ascii=False), encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "episode.synced.srt"
    result = pipeline.sync_episode(
        source, audio, output, tmp_path / "work", providers_path=providers, no_llm=no_llm,
        style_profile=StyleProfile(fps=30, min_cue_dur=0.5),
    )
    return parse_srt_text(output.read_text(encoding="utf-8")), result.report["flags"]


_CLARO_WORDS = [
    ("resolver", 3.08, 3.42), ("isso", 3.52, 3.72), ("dentro", 3.92, 4.12), ("da", 4.14, 4.24), ("empresa.", 4.32, 4.80),
    ("É", 5.36, 5.44), ("claro", 5.52, 5.66), ("que", 5.70, 5.78), ("a", 5.82, 5.86), ("empresa", 5.92, 6.14),
    ("está", 6.18, 6.32), ("disposta.", 6.36, 6.80),
]


def test_unspoken_source_cue_repeating_the_overlapping_cue_is_removed(tmp_path):
    # corpus.md B2 (ep17 cues 380/381): the script says "Claro," and then
    # "É claro que ..."; the actor says the phrase once. The cue without audio
    # stayed at source timing inside the cue that shows the same word.
    srt = (
        "1\n00:00:02,470 --> 00:00:04,600\nresolver isso dentro da empresa.\n\n"
        "2\n00:00:05,400 --> 00:00:05,990\nClaro,\n\n"
        "3\n00:00:06,320 --> 00:00:07,800\nÉ claro que a empresa está disposta.\n"
    )

    cues, flags = _sync(tmp_path, srt, _CLARO_WORDS)

    assert [cue.plain_text for cue in cues] == ["resolver isso dentro da empresa.", "É claro que a empresa está disposta."]
    merged = [flag for flag in flags if flag["kind"] == "duplicate_cue_merged"]
    assert [flag["cue_ids"] for flag in merged] == [[3, 2]]
    assert not [flag for flag in flags if flag["severity"] == "error"]
    assert not [flag for flag in flags if flag["cue_ids"] == [2]]


def test_unspoken_source_cue_with_other_words_is_kept_for_review(tmp_path):
    srt = (
        "1\n00:00:02,470 --> 00:00:04,600\nresolver isso dentro da empresa.\n\n"
        "2\n00:00:05,400 --> 00:00:05,990\nTalvez,\n\n"
        "3\n00:00:06,320 --> 00:00:07,800\nÉ claro que a empresa está disposta.\n"
    )

    cues, flags = _sync(tmp_path, srt, _CLARO_WORDS)

    assert [cue.plain_text for cue in cues] == [
        "resolver isso dentro da empresa.", "É claro que a empresa está disposta.", "Talvez,",
    ] or [cue.plain_text for cue in cues][1:] == ["Talvez,", "É claro que a empresa está disposta."]
    assert "duplicate_cue_merged" not in [flag["kind"] for flag in flags]
