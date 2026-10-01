from __future__ import annotations

import copy

import pytest

from dubsync.asr_crosscheck import (
    classify_spans,
    compare_word_streams,
    enforce_crosscheck_decisions,
    preaccepted_decision,
)
from dubsync.models import AdjudicationDecision, DivergenceSpan, Word


def words(text, *, offset=0.0, spacing=0.8):
    return [Word(text=token, start=offset + i * spacing, end=offset + i * spacing + 0.3)
            for i, token in enumerate(text.split())]


def span(source, audio, indices, *, case_id="case"):
    return DivergenceSpan(case_id=case_id, cue_ids=[1], srt_text=source, asr_text=audio,
                          asr_word_indices=indices)


def check(source, audio, primary, secondary, indices):
    item = span(source, audio, indices)
    agreement = compare_word_streams(primary, secondary)
    return item, classify_spans([item], agreement)[0]


def test_both_models_agree_on_exact_primary_wording_without_mutating_times():
    primary, secondary = words("Die Katze kommt."), words("Die KATZE kommt!", offset=0.04)
    before = copy.deepcopy((primary, secondary))
    item, evidence = check("Hund rennt", "Katze kommt.", primary, secondary, [1, 2])
    assert evidence.label == "both_agree"
    decision = preaccepted_decision(item, evidence)
    assert decision.verdict == "use_audio"
    assert decision.final_text == "Katze kommt."
    assert decision.heard_text == "Katze kommt."
    assert decision.evidence == "heard_clearly"
    assert (primary, secondary) == before


@pytest.mark.parametrize("primary_text,secondary_text", [("e", "é"), ("eine", "einen"), ("15", "fünfzehn")])
def test_semantic_or_number_aliases_are_not_exact_agreement(primary_text, secondary_text):
    item, evidence = check("source", primary_text, words(primary_text), words(secondary_text), [0])
    assert evidence.label != "both_agree"
    assert preaccepted_decision(item, evidence) is None


def test_repeated_nearby_words_remain_ambiguous():
    primary = words("Nein", offset=1)
    secondary = words("Nein nein", offset=0.9, spacing=0.2)
    item, evidence = check("Ja", "Nein", primary, secondary, [0])
    assert evidence.label == "ambiguous"
    assert preaccepted_decision(item, evidence) is None


def test_distant_repeated_phrase_does_not_corroborate():
    item, evidence = check("Ja", "Nein", words("Nein"), words("Nein", offset=8), [0])
    assert evidence.label != "both_agree"
    assert preaccepted_decision(item, evidence) is None


def test_stretched_secondary_word_does_not_corroborate():
    item, evidence = check("Ja", "Nein", words("Nein"), [Word(text="Nein", start=0, end=8)], [0])
    assert evidence.label != "both_agree"
    assert preaccepted_decision(item, evidence) is None


def test_swapped_order_does_not_corroborate_individual_words():
    primary = words("rot blau", spacing=0.2)
    secondary = words("blau rot", spacing=0.2)
    item, evidence = check("grün", "rot", primary, secondary, [0])
    assert evidence.label == "ambiguous"
    assert preaccepted_decision(item, evidence) is None


def test_secondary_extra_words_prevent_phrase_preacceptance():
    primary = [Word(text="Ich", start=0, end=0.2), Word(text="komme", start=1, end=1.3)]
    secondary = [primary[0], Word(text="nicht", start=0.5, end=0.7), primary[1]]
    item, evidence = check("Wir kommen", "Ich komme", primary, secondary, [0, 1])
    assert evidence.label != "both_agree"
    assert preaccepted_decision(item, evidence) is None


def test_asr_text_must_describe_the_owned_primary_words():
    item, evidence = check("Hund", "Katze", words("Die Maus kommt"), words("Die Maus kommt"), [1])
    assert evidence.label == "ambiguous"
    assert preaccepted_decision(item, evidence) is None


def test_noncontiguous_owned_words_do_not_preaccept():
    item, evidence = check("source", "Ich komme", words("Ich nicht komme"), words("Ich nicht komme"), [0, 2])
    assert evidence.label == "ambiguous"
    assert preaccepted_decision(item, evidence) is None


def test_different_token_grouping_can_agree_without_changing_primary_indices():
    primary = [Word(text="東京", start=0, end=0.4)]
    secondary = [Word(text="東", start=0.02, end=0.2), Word(text="京", start=0.22, end=0.42)]
    item, evidence = check("大阪", "東京", primary, secondary, [0])
    assert evidence.label == "both_agree"
    assert evidence.primary_word_indices == (0,)
    assert evidence.secondary_word_indices == (0, 1)


@pytest.mark.parametrize("evidence", [None, "heard_unclear", "heard_clearly"])
def test_secondary_script_agreement_requires_explicit_clear_audio_evidence(evidence):
    item, corroboration = check("Hund", "Katze", words("Die Katze kommt"), words("Die Hund kommt"), [1])
    assert corroboration.label == "secondary_matches_script"
    kwargs = {} if evidence is None else {"evidence": evidence, "heard_text": "Katze"}
    decision = AdjudicationDecision(case_id=item.case_id, verdict="use_audio", final_text="Katze",
                                    confidence=0.99, reason="reviewed", **kwargs)
    actual = enforce_crosscheck_decisions([item], [decision], [corroboration])[0]
    assert actual.verdict == ("use_audio" if evidence == "heard_clearly" else "keep_srt")


@pytest.mark.parametrize("audio_evidence", [None, "heard_unclear", "heard_clearly"])
def test_primary_only_insertion_requires_explicit_clear_audio_review(audio_evidence):
    primary = words("Hallo ach Welt")
    secondary = [primary[0], primary[2]]
    item, evidence = check("", "ach", primary, secondary, [1])
    assert evidence.label == "primary_only_insertion"
    assert preaccepted_decision(item, evidence) is None
    kwargs = {} if audio_evidence is None else {"evidence": audio_evidence, "heard_text": "ach"}
    decision = AdjudicationDecision(case_id=item.case_id, verdict="use_audio", final_text="ach",
                                    confidence=1, reason="reviewed", **kwargs)
    actual = enforce_crosscheck_decisions([item], [decision], [evidence])[0]
    assert actual.verdict == ("use_audio" if audio_evidence == "heard_clearly" else "keep_srt")
    assert actual.final_text == ("ach" if audio_evidence == "heard_clearly" else "")
    assert actual.confidence == (1 if audio_evidence == "heard_clearly" else 0)


def test_supported_insertion_is_still_reviewed_not_preaccepted():
    item, evidence = check("", "ach", words("ach"), words("ach"), [0])
    assert evidence.label == "both_agree"
    assert preaccepted_decision(item, evidence) is None


def test_missing_decision_for_primary_only_insertion_gets_an_explicit_hold():
    item, evidence = check("", "ach", words("ach"), [], [0])
    actual = enforce_crosscheck_decisions([item], [], [evidence])
    assert len(actual) == 1
    assert actual[0].verdict == "keep_srt"


def test_empty_or_invalid_index_spans_cannot_be_preaccepted():
    agreement = compare_word_streams(words("Hallo"), words("Hallo"))
    for indices in ([], [-1], [9], [0, 0]):
        item = span("Tschüss", "Hallo", indices)
        evidence = classify_spans([item], agreement)[0]
        assert evidence.label == "ambiguous"
        assert preaccepted_decision(item, evidence) is None


@pytest.mark.parametrize("source,audio", [
    ("Damien", "Damian"), ("Zang", "Zhang"), ("rápida", "rápido"),
    ("Fico brava", "Fica bravo"), ("Visite Damien hoje", "Visite Damian hoje"),
    ("Não volto hoje", "Eu volto hoje"), ("Traga 12 flores", "Traga 13 flores"),
])
def test_agreed_names_inflections_negations_and_quantities_still_need_audio_review(source, audio):
    primary = words(audio)
    item, evidence = check(source, audio, primary, primary, list(range(len(primary))))
    assert evidence.label == "both_agree"
    assert preaccepted_decision(item, evidence) is None


def test_corroborated_wording_across_cues_still_needs_ownership_review():
    primary = words("Vou falar amanhã")
    item, evidence = check("Quero dizer isso depois", "Vou falar amanhã", primary, primary, [0, 1, 2])
    item = item.model_copy(update={"cue_ids": [1, 2]})
    assert preaccepted_decision(item, evidence) is None


def test_two_models_hearing_different_interjections_is_a_reviewable_conflict_not_primary_only():
    primary = words("Hallo hum Welt")
    secondary = words("Hallo hmmm Welt")
    item, evidence = check("", "hum", primary, secondary, [1])
    assert evidence.label == "conflict"
    decision = AdjudicationDecision(case_id=item.case_id, verdict="use_audio", final_text="hum",
                                    evidence="heard_clearly", heard_text="hum", confidence=1, reason="reviewed")
    assert enforce_crosscheck_decisions([item], [decision], [evidence]) == [decision]


def test_partial_insertion_support_remains_primary_only():
    primary = words("Hallo schöne neue Welt")
    secondary = [primary[0], primary[1], primary[3]]
    item, evidence = check("", "schöne neue", primary, secondary, [1, 2])
    assert evidence.label == "primary_only_insertion"


def test_crosscheck_does_not_override_an_existing_source_preservation_hold():
    item, evidence = check("", "ach", words("ach"), [], [0])
    decision = AdjudicationDecision(case_id=item.case_id, verdict="keep_srt", final_text="", confidence=1,
                                    reason="The missing source region is protected.")
    assert enforce_crosscheck_decisions([item], [decision], [evidence]) == [decision]


@pytest.mark.parametrize("language", [None, "ja", "en"])
@pytest.mark.parametrize("source,audio", [("大阪", "東京"), ("ハナ", "ユミ"), ("去北京", "去上海")])
def test_character_tokens_do_not_turn_one_proper_noun_into_a_multiword_preaccept(source, audio, language):
    primary = [Word(text=audio, start=1, end=1.5)]
    item, evidence = check(source, audio, primary, primary, [0])
    assert evidence.label == "both_agree"
    assert preaccepted_decision(item, evidence, language=language) is None
    reviewed = AdjudicationDecision(case_id=item.case_id, verdict="use_audio", final_text=audio,
                                    confidence=1, reason="Audio reviewed", evidence="heard_clearly", heard_text=audio)
    assert enforce_crosscheck_decisions([item], [reviewed], [evidence]) == [reviewed]


def test_preaccept_support_depends_on_text_evidence_not_the_language_picker():
    primary = words("Katze kommt")
    item, evidence = check("Hund rennt", "Katze kommt", primary, primary, [0, 1])
    assert preaccepted_decision(item, evidence, language="ja") is not None
