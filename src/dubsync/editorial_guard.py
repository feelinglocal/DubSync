from __future__ import annotations

from collections import Counter
import re
import unicodedata

from .models import AdjudicationDecision, Cue, DivergenceSpan, QCFlag
from .tokenize import alphanumeric_signature


class EditorialGuardError(ValueError):
    pass


_ALWAYS_QUOTATION_MARKS = frozenset('"\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a\u201a')
_CONTEXTUAL_SINGLE_QUOTATION_MARKS = frozenset("'\u2018\u2019")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]*>")
_MASKED_WORD_RE = re.compile(r"[\w]*[*][\w*]*", re.UNICODE)


def validate_adjudication_editorial_contract(
    span: DivergenceSpan,
    decision: AdjudicationDecision,
    *,
    allow_word_change: bool,
    source_context: str | None = None,
    applied_text: str | None = None,
) -> None:
    """Check an approved wording against the source span it replaces.

    The span text is built from alignment tokens and never contains
    punctuation. ``source_context`` is the authored cue text of those tokens
    with the punctuation directly around them: a quotation mark found there
    is one the source already has, so the wording may repeat it.
    ``applied_text`` is the part of the wording that replaces the tokens once
    repeated cue context is removed. The edited cue is checked as a whole
    afterwards; here only marks without a source counterpart are refused.
    """
    if source_context is None:
        validate_editorial_text(span.srt_text, decision.final_text, allow_word_change=allow_word_change)
        return
    validate_editorial_text(
        span.srt_text, decision.final_text, allow_word_change=allow_word_change, check_quotation_marks=False,
    )
    applied = decision.final_text if applied_text is None else applied_text
    if _signature_additions(_quotation_mark_signature(source_context), _quotation_mark_signature(applied)):
        raise EditorialGuardError("quotation mark signature changed during adjudication")


def validate_editorial_text(
    before: str,
    after: str,
    *,
    allow_word_change: bool,
    check_quotation_marks: bool = True,
    allow_removed_quotations: bool = False,
) -> None:
    """Refuse editorial changes that word adjudication has no authority for.

    ``allow_removed_quotations`` accepts a result that lost complete
    quotations (an approved deletion of the quoted words); a single missing
    or any added mark is still a changed signature.
    """
    if not allow_word_change and alphanumeric_signature(before) != alphanumeric_signature(after):
        raise EditorialGuardError("alphanumeric content changed without word-change authority")
    if check_quotation_marks:
        source_marks, edited_marks = _quotation_mark_signature(before), _quotation_mark_signature(after)
        if source_marks != edited_marks and not (
            allow_removed_quotations and _only_whole_quotations_removed(source_marks, edited_marks)
        ):
            raise EditorialGuardError("quotation mark signature changed during adjudication")
    if _html_tag_signature(before) != _html_tag_signature(after):
        raise EditorialGuardError("markup signature changed during adjudication")
    if _masked_word_signature(before) != _masked_word_signature(after):
        raise EditorialGuardError("censor-mask signature changed during adjudication")


def episode_editorial_addition_flags(
    source_cues: list[Cue],
    output_cues: list[Cue],
) -> list[QCFlag]:
    source_text = "\n".join(cue.text for cue in source_cues)
    output_text = "\n".join(cue.text for cue in output_cues)
    additions = {
        "quotation marks": _signature_additions(
            _quotation_mark_signature(source_text),
            _quotation_mark_signature(output_text),
        ),
        "markup": _signature_additions(
            _html_tag_signature(source_text),
            _html_tag_signature(output_text),
        ),
        "censor masks": _signature_additions(
            _masked_word_signature(source_text),
            _masked_word_signature(output_text),
        ),
    }
    detected = [label for label, values in additions.items() if values]
    if not detected:
        return []
    return [
        QCFlag(
            kind="editorial_signature_unexplained",
            cue_ids=[],
            message=f"Unexplained editorial additions detected: {', '.join(detected)}.",
            severity="error",
            start=(output_cues[0].start_ms / 1000.0) if output_cues else None,
            end=(output_cues[-1].end_ms / 1000.0) if output_cues else None,
        )
    ]


def _signature_additions(before: tuple[str, ...], after: tuple[str, ...]) -> tuple[str, ...]:
    return tuple((Counter(after) - Counter(before)).elements())


def _only_whole_quotations_removed(before: tuple[str, ...], after: tuple[str, ...]) -> bool:
    """The edited marks are the source marks minus complete opening/closing pairs."""
    if len(before) % 2 or len(after) % 2:
        return False
    pairs = list(zip(before[0::2], before[1::2]))
    remaining = iter(pairs)
    return all(pair in remaining for pair in zip(after[0::2], after[1::2]))


def _quotation_mark_signature(text: str) -> tuple[str, ...]:
    signature: list[str] = []
    for index, character in enumerate(text):
        if character in _ALWAYS_QUOTATION_MARKS:
            signature.append(character)
            continue
        if character not in _CONTEXTUAL_SINGLE_QUOTATION_MARKS:
            continue
        previous = text[index - 1] if index > 0 else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if _is_word_character(previous) and _is_word_character(following):
            continue
        signature.append(character)
    return tuple(signature)


def _html_tag_signature(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _HTML_TAG_RE.finditer(text))


def _masked_word_signature(text: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFC", match.group(0)).casefold()
        for match in _MASKED_WORD_RE.finditer(text)
    )


def _is_word_character(character: str) -> bool:
    if not character:
        return False
    category = unicodedata.category(character)
    return character == "_" or category[0] in {"L", "M", "N"}
