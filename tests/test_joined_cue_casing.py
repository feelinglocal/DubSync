"""Casing at proved joins preserves words, acoustic ownership and authored evidence."""
from __future__ import annotations

import pytest

from dubsync.changes import apply_adjudication_decisions
from dubsync.cue_segmentation import join_one_letter_residues, settle_collapsed_generated_adlibs
from dubsync.models import AdjudicationDecision, AlignmentResult, Cue, DivergenceSpan, TokenMatch, Word
from dubsync.pipeline import _alignment_with_decision_words
from dubsync.punctuation import validate_punctuation_only
from dubsync.recue import rebuild_cues
from dubsync.style_profile import StyleProfile
from dubsync.subtitle_annotations import speech_text_for_alignment
from dubsync.text_metrics import join_word_texts, token_texts
from dubsync.tokenize import alphanumeric_signature


_PROFILE = StyleProfile(max_chars_per_line=100, max_lines_per_cue=2, min_cue_dur=0.1,
                        lead_in_ms=0, tail_ms=0)


def _prefix_case(text, prefix, first_asr, *, context=None):
    # The lexical casing in the regression cases comes from cached ep11,
    # ep17 and testlong-1 transcripts; times here are a small synthetic window.
    target = Cue(index=10, start_ms=10000, end_ms=12000, lines=text.splitlines())
    cues = [target] if context is None else [
        Cue(index=5, start_ms=8000, end_ms=9000, lines=[context]), target,
    ]
    prefix_tokens = token_texts(prefix)
    source_tokens = token_texts(speech_text_for_alignment(target))
    spoken = [*prefix_tokens, first_asr, *source_tokens[1:]]
    words = [Word(text=word, start=10 + index * 0.15, end=10.1 + index * 0.15,
                  confidence=None) for index, word in enumerate(spoken)]
    count = len(prefix_tokens)
    offset = len(alphanumeric_signature(context or ""))
    matches = [TokenMatch(cue_id=10, srt_token_index=offset + position,
                          asr_word_index=count + position, score=1.0)
               for position in range(len(source_tokens))]
    span = DivergenceSpan(case_id="prefix", cue_ids=[], srt_text="", asr_text=prefix,
                          asr_word_indices=list(range(count)), start=words[0].start,
                          end=words[count - 1].end, right_anchor_cue_id=10,
                          right_anchor_start=words[count].start)
    decision = AdjudicationDecision(case_id="prefix", verdict="use_audio", final_text=prefix,
                                    confidence=0.98, reason="approved spoken prefix")
    alignment = AlignmentResult(token_matches=matches, divergence_spans=[span],
                                cue_word_indices={10: list(range(count, len(words)))})
    return cues, words, span, decision, alignment


def _apply_prefix(case, *, matches=None, words_available=True, extra_spans=(), extra_decisions=()):
    cues, words, span, decision, alignment = case
    return apply_adjudication_decisions(
        cues, [span, *extra_spans], [decision, *extra_decisions], _PROFILE,
        adlib_cue_ids_by_case={"prefix": 10}, words=words if words_available else None,
        token_matches=alignment.token_matches if matches is None else matches,
    )


@pytest.mark.parametrize("text,prefix,first_asr,expected", [
    ("Vou chamar a polícia.", "Eu", "vou", "Eu vou chamar a polícia."),
    ("Olha pra frente.", "Cuidado,", "olha", "Cuidado, olha pra frente."),
    ("A checagem", "sem", "a", "Sem a checagem"),
    ("Vou chamar a polícia.", "eu", "vou", "Eu vou chamar a polícia."),
])
def test_proved_prefix_recases_only_the_new_sentence_boundary(text, prefix, first_asr, expected):
    case = _prefix_case(text, prefix, first_asr)
    cues, words, span, decision, alignment = case
    before = ([cue.model_dump() for cue in cues], [word.model_dump() for word in words], alignment.model_dump())
    changed, flags = _apply_prefix(case)

    assert changed[0].plain_text == expected
    assert (changed[0].start_ms, changed[0].end_ms) == (10000, 12000)
    raw_join = join_word_texts([prefix, text])
    assert alphanumeric_signature(expected) == alphanumeric_signature(raw_join)
    assert validate_punctuation_only(raw_join, expected) == expected
    assert [(flag.kind, flag.cue_ids, flag.new_text) for flag in flags] == [("text_changed", [10], expected)]

    mapped = _alignment_with_decision_words(
        alignment, [decision], [span], {"prefix": 10}, source_cues=cues, words=words,
    )
    assert mapped.cue_word_indices == {10: list(range(len(words)))}
    rebuilt, timing_flags = rebuild_cues(changed, words, mapped, _PROFILE)
    assert (rebuilt[0].start_ms, rebuilt[0].end_ms) == (
        _PROFILE.snap_floor(words[0].start * 1000), _PROFILE.snap_ceil(words[-1].end * 1000),
    )
    assert not [flag for flag in timing_flags if "held" in flag.kind]
    assert before == ([cue.model_dump() for cue in cues], [word.model_dump() for word in words], alignment.model_dump())


@pytest.mark.parametrize("text,prefix,first_asr,expected,context", [
    ("Vou chamar a polícia.", "Eu", "Vou", "Eu Vou chamar a polícia.", None),
    ("Haus bleibt stehen.", "Das", "Haus", "Das Haus bleibt stehen.", None),
    ("Haus bleibt stehen.", "Das", "haus", "Das Haus bleibt stehen.", "Unser Haus ist groß."),
    ("Anna chegou.", "E", "anna", "E Anna chegou.", "Eu falei com Anna."),
    ("McDonald chegou.", "O", "mcdonald", "O McDonald chegou.", None),
    ("NASA chegou.", "A", "nasa", "A NASA chegou.", None),
    ("I agree.", "Yes,", "i", "Yes, I agree.", None),
    ("Vou chamar a polícia.", "Eu.", "vou", "Eu. Vou chamar a polícia.", None),
    ("Vou chamar a polícia.", "Eu!", "vou", "Eu! Vou chamar a polícia.", None),
    ("Vou chamar a polícia.", "Eu?", "vou", "Eu? Vou chamar a polícia.", None),
    ("vou chamar a polícia.", "eu", "vou", "eu vou chamar a polícia.", None),
    ('"Olha pra frente."', "Cuidado,", "olha", '" Cuidado, Olha pra frente."', None),
    ("<i>Olha pra frente.</i>", "Cuidado,", "olha", "<i> Cuidado, Olha pra frente.</i>", None),
    ("- Olha pra frente.", "Cuidado,", "olha", "- Cuidado, Olha pra frente.", None),
    ("[ON SCREEN]\nOlha pra frente.", "Cuidado,", "olha", "[ON SCREEN] Cuidado, Olha pra frente.", None),
])
def test_prefix_preserves_capitalization_with_protected_or_continuation_evidence(
    text, prefix, first_asr, expected, context,
):
    changed, flags = _apply_prefix(_prefix_case(text, prefix, first_asr, context=context))
    assert changed[-1].plain_text == expected
    assert flags[0].new_text.replace("\n", " ") == expected


def test_source_sentence_boundary_after_a_dialogue_dash_is_not_name_evidence():
    # Real ep11 source 507 starts a new actor's sentence after the full stop;
    # that capital must not prevent source 228's "Olha" becoming a continuation.
    changed, _ = _apply_prefix(_prefix_case(
        "Olha pra frente.", "Cuidado,", "olha", context="- Luke, foto. - Olha para a câmera.",
    ))
    assert changed[-1].plain_text == "Cuidado, olha pra frente."


@pytest.mark.parametrize("variant", ["no_words", "no_matches", "wrong_word", "wrong_cue", "fuzzy", "ambiguous"])
def test_prefix_never_searches_elsewhere_for_missing_or_ambiguous_first_token_evidence(variant):
    case = _prefix_case("Vou chamar a polícia.", "Eu", "vou")
    cues, words, _, _, alignment = case
    matches = list(alignment.token_matches)
    # A lower-case copy elsewhere in the episode cannot replace the retained
    # source token's exact match.
    words.append(Word(text="vou", start=90, end=90.2))
    if variant == "no_matches":
        matches = matches[1:]
    elif variant == "wrong_word":
        matches[0] = matches[0].model_copy(update={"asr_word_index": 2})
    elif variant == "wrong_cue":
        matches[0] = matches[0].model_copy(update={"cue_id": 9})
    elif variant == "fuzzy":
        matches[0] = matches[0].model_copy(update={"score": 0.9})
    elif variant == "ambiguous":
        matches.append(matches[0].model_copy(update={"asr_word_index": len(words) - 1}))
    changed, _ = _apply_prefix(case, matches=matches, words_available=variant != "no_words")
    assert changed[0].plain_text == "Eu Vou chamar a polícia."
    assert cues[0].plain_text == "Vou chamar a polícia."


def test_prefix_does_not_recase_a_first_word_replaced_by_an_independent_accepted_edit():
    case = _prefix_case("Vou chamar a polícia.", "Eu", "vou")
    extra = DivergenceSpan(case_id="word", cue_ids=[10], srt_token_indices=[0],
                           srt_text="Vou", asr_text="Vamos", asr_word_indices=[1])
    decision = AdjudicationDecision(case_id="word", verdict="use_audio", final_text="Vamos",
                                    confidence=0.98, reason="independent accepted wording")
    changed, flags = _apply_prefix(case, extra_spans=[extra], extra_decisions=[decision])
    assert changed[0].plain_text == "Eu Vamos chamar a polícia."
    assert all(flag.new_text == changed[0].text for flag in flags)


def _residue_case(first_asr="ele", *, text="Ele ainda tentou me acalmar,", context=None, collapsed=False):
    prefix = Cue(index=198, start_ms=824000, end_ms=824001 if collapsed else 824100, lines=["e"])
    target = Cue(index=199, start_ms=824120, end_ms=825600, lines=[text])
    cues = [prefix, target] if context is None else [
        Cue(index=197, start_ms=822000, end_ms=823000, lines=[context]), prefix, target,
    ]
    # The E/ele lexical casing is the actual ep17 ASR evidence. The other
    # endpoints isolate the existing join rule without depending on the WAV.
    words = [Word(text="E", start=824, end=824.001 if collapsed else 824.1)]
    spoken = [first_asr, *token_texts(text)[1:]]
    words.extend(Word(text=word, start=824.12 + index * 0.2, end=824.24 + index * 0.2)
                 for index, word in enumerate(spoken))
    alignment = AlignmentResult(cue_word_indices={198: [0], 199: list(range(1, len(words)))})
    return cues, words, alignment


@pytest.mark.parametrize("first_asr,text,context,expected", [
    ("ele", "Ele ainda tentou me acalmar,", None, "E ele ainda tentou me acalmar,"),
    ("Ele", "Ele ainda tentou me acalmar,", None, "E Ele ainda tentou me acalmar,"),
    ("anna", "Anna chegou.", "Eu falei com Anna.", "E Anna chegou."),
    ("Haus", "Haus bleibt stehen.", None, "E Haus bleibt stehen."),
    ("mcdonald", "McDonald chegou.", None, "E McDonald chegou."),
])
def test_residue_join_uses_only_proved_asr_casing_and_preserves_its_existing_acoustic_union(
    first_asr, text, context, expected,
):
    cues, words, alignment = _residue_case(first_asr, text=text, context=context)
    before = ([cue.model_dump() for cue in cues], [word.model_dump() for word in words], alignment.model_dump())
    changed, mapped, flags, gone = join_one_letter_residues(
        cues, words, alignment, _PROFILE, residue_cue_ids={198},
    )
    merged = changed[-1]
    assert merged.plain_text == expected
    assert (merged.start_ms, merged.end_ms) == (824000, 825600)
    assert gone == {198}
    assert mapped.cue_word_indices == {199: list(range(len(words)))}
    assert alphanumeric_signature(expected) == alphanumeric_signature(join_word_texts(["e", text]))
    assert validate_punctuation_only(join_word_texts(["e", text]), expected) == expected
    assert [(flag.kind, flag.cue_ids, flag.new_text) for flag in flags] == [("text_changed", [199], expected)]
    assert before == ([cue.model_dump() for cue in cues], [word.model_dump() for word in words], alignment.model_dump())


@pytest.mark.parametrize("variant", ["missing", "ambiguous"])
def test_residue_join_preserves_old_initial_without_a_unique_exact_owned_word_window(variant):
    cues, words, alignment = _residue_case()
    indices = [] if variant == "missing" else [1, *range(1, len(words))]
    if variant == "ambiguous":
        words.append(words[1].model_copy(update={"start": 824.11, "end": 824.12}))
        indices = [len(words) - 1, *range(1, len(words))]
    alignment = alignment.model_copy(update={"cue_word_indices": {198: [0], 199: indices}})
    changed, _, _, _ = join_one_letter_residues(cues, words, alignment, _PROFILE, residue_cue_ids={198})
    assert changed[-1].plain_text == "E Ele ainda tentou me acalmar,"


def test_residue_join_preserves_a_quoted_sentence_initial():
    cues, words, alignment = _residue_case(text='"Ele ainda tentou me acalmar,"')
    changed, _, flags, gone = join_one_letter_residues(cues, words, alignment, _PROFILE, residue_cue_ids={198})
    assert changed[-1].plain_text == 'e "Ele ainda tentou me acalmar,"'
    assert gone == {198}
    assert flags[0].new_text == changed[-1].text


def test_collapsed_adlib_prepend_uses_the_same_boundary_rule_without_changing_its_timing_policy():
    cues, words, alignment = _residue_case(collapsed=True)
    changed, mapped, flags, gone = settle_collapsed_generated_adlibs(
        cues, words, alignment, _PROFILE, collapsed_cue_ids={198},
    )
    assert changed[0].plain_text == "E ele ainda tentou me acalmar,"
    assert (changed[0].start_ms, changed[0].end_ms) == (824120, 825600)
    assert mapped.cue_word_indices == {199: list(range(1, len(words)))}
    assert gone == {198}
    assert flags[0].new_text == changed[0].text


def test_unchanged_cues_keep_exact_authored_case_and_line_breaks():
    cue = Cue(index=10, start_ms=10000, end_ms=12000, lines=["Eu Vou chamar", "a polícia."])
    words = [Word(text="eu", start=10, end=10.1), Word(text="vou", start=10.15, end=10.25)]
    changed, flags = apply_adjudication_decisions([cue], [], [], _PROFILE, words=words)
    assert changed == [cue]
    assert changed[0].lines == ["Eu Vou chamar", "a polícia."]
    assert flags == []
