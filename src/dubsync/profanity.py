from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass

from .models import Cue, QCFlag


GERMAN_PROFANITY_RULESET_VERSION = "de-v1-2026-07-30"
LEXICON_VERSION = GERMAN_PROFANITY_RULESET_VERSION
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:\*+[^\W\d_]+)*", re.UNICODE)
_MASK_RE = re.compile(r"\*+")
_MASK_SUBSTITUTIONS = tuple("abcdefghijklmnopqrstuvwxyz") + ("ae", "oe", "ue", "ss")


@dataclass(frozen=True)
class GermanProfanityTerm:
    canonical: str
    forms: tuple[str, ...]
    prefixes: tuple[str, ...]
    mask_letters: str
    mask_occurrence: int = 0


@dataclass(frozen=True)
class _SourceMask:
    position: int
    cue_id: int
    canonical: str
    token: str


# This is an intentionally versioned, conservative production policy. The
# entries are whole-token families or strong vulgar roots; ambiguous everyday
# substrings such as "Mist" in "Mistel" are not treated as prefixes.
GERMAN_PROFANITY_TERMS: tuple[GermanProfanityTerm, ...] = (
    GermanProfanityTerm(
        canonical="arschloch",
        forms=("arschloch", "arschloecher"),
        prefixes=("arschloch",),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="verdammt",
        forms=("verdammt", "verdammte", "verdammter", "verdammtes", "verdammten", "verdammtem"),
        prefixes=(),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="mist",
        forms=("mist", "mistkerl", "mistkerle", "miststueck", "miststuecke", "mistvieh"),
        prefixes=(),
        mask_letters="i",
    ),
    GermanProfanityTerm(
        canonical="scheisse",
        forms=("scheisse", "scheiss"),
        prefixes=("scheiss",),
        mask_letters="e",
    ),
    GermanProfanityTerm(
        canonical="bastard",
        forms=("bastard", "bastarde", "bastarden", "bastards"),
        prefixes=(),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="idiot",
        forms=("idiot", "idiotin", "idioten", "idiotinnen", "vollidiot", "vollidiotin", "vollidioten"),
        prefixes=(),
        mask_letters="i",
        mask_occurrence=1,
    ),
    GermanProfanityTerm(
        canonical="dummkopf",
        forms=("dummkopf", "dummkoepfe", "dummkoepfen"),
        prefixes=(),
        mask_letters="u",
    ),
    GermanProfanityTerm(
        canonical="verarschen",
        forms=("verarsch", "verarsche", "verarschen", "verarscht", "verarschte", "verarschung"),
        prefixes=("verarsch",),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="beschissen",
        forms=("beschissen",),
        prefixes=("beschiss",),
        mask_letters="i",
    ),
    GermanProfanityTerm(
        canonical="arsch",
        forms=("arsch", "aersche", "aerschen"),
        prefixes=("arsch",),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="ficken",
        forms=("fick", "ficke", "ficken", "fickst", "fickt", "fickte", "fickten"),
        prefixes=("fick", "gefick", "verfick"),
        mask_letters="i",
    ),
    GermanProfanityTerm(
        canonical="fotze",
        forms=("fotze", "fotzen"),
        prefixes=("fotz",),
        mask_letters="o",
    ),
    GermanProfanityTerm(
        canonical="hurensohn",
        forms=("hurensohn", "hurensoehne", "hurensoehnen"),
        prefixes=("hurensohn",),
        mask_letters="u",
    ),
    GermanProfanityTerm(
        canonical="hure",
        forms=("hure", "huren"),
        prefixes=(),
        mask_letters="u",
    ),
    GermanProfanityTerm(
        canonical="wichser",
        forms=("wichser", "wichsern"),
        prefixes=("wichs",),
        mask_letters="i",
    ),
    GermanProfanityTerm(
        canonical="kacke",
        forms=("kacke",),
        prefixes=("kack",),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="pisse",
        forms=("pisse", "pisser", "pissern"),
        prefixes=("piss", "verpiss", "angepiss"),
        mask_letters="i",
    ),
    GermanProfanityTerm(
        canonical="depp",
        forms=("depp", "deppen"),
        prefixes=(),
        mask_letters="e",
    ),
    GermanProfanityTerm(
        canonical="trottel",
        forms=("trottel", "trotteln"),
        prefixes=(),
        mask_letters="o",
    ),
    GermanProfanityTerm(
        canonical="bloedmann",
        forms=("bloedmann", "bloedmaenner", "bloedmaennern"),
        prefixes=(),
        mask_letters="o\u00f6",
    ),
    GermanProfanityTerm(
        canonical="schlampe",
        forms=("schlampe", "schlampen"),
        prefixes=(),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="nutte",
        forms=("nutte", "nutten"),
        prefixes=(),
        mask_letters="u",
    ),
    GermanProfanityTerm(
        canonical="fresse",
        forms=("fresse",),
        prefixes=(),
        mask_letters="e",
    ),
    GermanProfanityTerm(
        canonical="vollpfosten",
        forms=("vollpfosten",),
        prefixes=(),
        mask_letters="o",
    ),
    GermanProfanityTerm(
        canonical="schwachkopf",
        forms=("schwachkopf", "schwachkoepfe"),
        prefixes=(),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="saftsack",
        forms=("saftsack", "saftsaecke"),
        prefixes=(),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="schwanzlutscher",
        forms=("schwanzlutscher",),
        prefixes=(),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="armleuchter",
        forms=("armleuchter",),
        prefixes=(),
        mask_letters="e",
    ),
    GermanProfanityTerm(
        canonical="dumpfbacke",
        forms=("dumpfbacke", "dumpfbacken"),
        prefixes=(),
        mask_letters="u",
    ),
    GermanProfanityTerm(
        canonical="doedel",
        forms=("doedel",),
        prefixes=(),
        mask_letters="o\u00f6",
    ),
    GermanProfanityTerm(
        canonical="einfaltspinsel",
        forms=("einfaltspinsel",),
        prefixes=(),
        mask_letters="a",
    ),
    GermanProfanityTerm(
        canonical="hornochse",
        forms=("hornochse", "hornochsen"),
        prefixes=(),
        mask_letters="o",
    ),
    GermanProfanityTerm(
        canonical="schweinebacke",
        forms=("schweinebacke", "schweinebacken"),
        prefixes=(),
        mask_letters="e",
    ),
    GermanProfanityTerm(
        canonical="sackratte",
        forms=("sackratte", "sackratten"),
        prefixes=(),
        mask_letters="a",
    ),
)


def canonicalize_german_profanity_token(value: str) -> str | None:
    if not value:
        return None
    normalized = _normalize_form(value)
    if "*" not in normalized:
        term = _match_uncensored_normalized(normalized)
        return term.canonical if term is not None else None
    term = _match_masked_normalized(normalized)
    return term.canonical if term is not None else None


def censor_german_profanity_text(text: str, source_text: str | None = None) -> str:
    source_masks = _collect_source_masks(
        [Cue(index=0, start_ms=0, end_ms=1, lines=[source_text])] if source_text else []
    )
    reservations = _output_reservations(
        [Cue(index=0, start_ms=0, end_ms=1, lines=[text])]
    )
    censored, _, _ = _censor_text(
        text,
        cue_id=0,
        source_masks=source_masks,
        used=frozenset(),
        reservations=reservations,
    )
    return censored


def apply_german_profanity_censorship(
    cues: list[Cue],
    source_cues: list[Cue] | None = None,
) -> tuple[list[Cue], list[QCFlag]]:
    source_masks = _collect_source_masks(source_cues or [])
    reservations = _output_reservations(cues)
    used: frozenset[int] = frozenset()
    updated: list[Cue] = []
    flags: list[QCFlag] = []

    for cue in cues:
        next_lines: list[str] = []
        cue_changes: list[tuple[str, str]] = []
        for line in cue.lines:
            next_line, line_changes, used = _censor_text(
                line,
                cue_id=cue.index,
                source_masks=source_masks,
                used=used,
                reservations=reservations,
            )
            next_lines.append(next_line)
            cue_changes.extend(line_changes)
        if not cue_changes:
            updated.append(cue)
            continue
        updated.append(cue.with_lines(next_lines))
        flags.extend(
            QCFlag(
                kind="german_profanity_censored",
                cue_ids=[cue.index],
                message=(
                    "German profanity was masked using "
                    f"{GERMAN_PROFANITY_RULESET_VERSION}; exact source masks are preserved when available."
                ),
                severity="info",
                old_text=old,
                new_text=new,
                start=cue.start_ms / 1000.0,
                end=cue.end_ms / 1000.0,
            )
            for old, new in cue_changes
        )

    return updated, flags


def censor_german_profanity_flags(
    flags: list[QCFlag],
    source_cues: list[Cue] | None = None,
) -> list[QCFlag]:
    source_text = "\n".join(cue.text for cue in (source_cues or [])) or None
    return [
        flag.model_copy(
            update={
                "message": censor_german_profanity_text(flag.message, source_text),
                "old_text": (
                    censor_german_profanity_text(flag.old_text, source_text)
                    if flag.old_text is not None
                    else None
                ),
                "new_text": (
                    censor_german_profanity_text(flag.new_text, source_text)
                    if flag.new_text is not None
                    else None
                ),
            }
        )
        for flag in flags
    ]


def _censor_text(
    text: str,
    *,
    cue_id: int,
    source_masks: tuple[_SourceMask, ...],
    used: frozenset[int],
    reservations: frozenset[tuple[int, str]],
) -> tuple[str, list[tuple[str, str]], frozenset[int]]:
    changes: list[tuple[int, int, str, str]] = []
    next_used = used
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        canonical = canonicalize_german_profanity_token(token)
        if canonical is None:
            continue
        source_mask, next_used = _take_source_mask(
            source_masks,
            used=next_used,
            cue_id=cue_id,
            canonical=canonical,
            reservations=reservations,
        )
        if source_mask is not None:
            replacement = source_mask.token
        elif "*" in token:
            replacement = token
        else:
            term = _TERM_BY_CANONICAL[canonical]
            replacement = _replacement_for_token(token, term)
        if replacement != token:
            changes.append((match.start(), match.end(), token, replacement))

    if not changes:
        return text, [], next_used
    pieces: list[str] = []
    cursor = 0
    for start, end, old, new in changes:
        pieces.extend((text[cursor:start], new))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), [(old, new) for _, _, old, new in changes], next_used


def _collect_source_masks(cues: list[Cue]) -> tuple[_SourceMask, ...]:
    masks: list[_SourceMask] = []
    for cue in cues:
        for match in _TOKEN_RE.finditer(cue.text):
            token = match.group(0)
            if "*" not in token:
                continue
            canonical = canonicalize_german_profanity_token(token)
            if canonical is None:
                continue
            masks.append(
                _SourceMask(
                    position=len(masks),
                    cue_id=cue.index,
                    canonical=canonical,
                    token=token,
                )
            )
    return tuple(masks)


def _output_reservations(cues: list[Cue]) -> frozenset[tuple[int, str]]:
    return frozenset(
        (cue.index, canonical)
        for cue in cues
        for match in _TOKEN_RE.finditer(cue.text)
        if (canonical := canonicalize_german_profanity_token(match.group(0))) is not None
    )


def _take_source_mask(
    source_masks: tuple[_SourceMask, ...],
    *,
    used: frozenset[int],
    cue_id: int,
    canonical: str,
    reservations: frozenset[tuple[int, str]],
) -> tuple[_SourceMask | None, frozenset[int]]:
    same_cue = next(
        (
            item
            for item in source_masks
            if item.position not in used and item.cue_id == cue_id and item.canonical == canonical
        ),
        None,
    )
    selected = same_cue or next(
        (
            item
            for item in source_masks
            if (
                item.position not in used
                and item.canonical == canonical
                and (item.cue_id, canonical) not in reservations
            )
        ),
        None,
    )
    if selected is None:
        return None, used
    return selected, used | {selected.position}


def _match_uncensored_normalized(normalized: str) -> GermanProfanityTerm | None:
    exact = _FORM_TO_TERM.get(normalized)
    if exact is not None:
        return exact
    return next(
        (
            term
            for prefix, term in _ORDERED_PREFIXES
            if normalized.startswith(prefix) and len(normalized) >= len(prefix)
        ),
        None,
    )


def _match_masked_normalized(normalized: str) -> GermanProfanityTerm | None:
    groups = _MASK_RE.findall(normalized)
    if not groups or len(groups) > 2:
        return None
    matches: dict[str, GermanProfanityTerm] = {}
    for substitutions in itertools.product(_MASK_SUBSTITUTIONS, repeat=len(groups)):
        candidate = normalized
        for substitution in substitutions:
            candidate = _MASK_RE.sub(substitution, candidate, count=1)
        term = _match_uncensored_normalized(candidate)
        if term is not None:
            matches[term.canonical] = term
    if len(matches) != 1:
        return None
    return next(iter(matches.values()))


def _replacement_for_token(token: str, term: GermanProfanityTerm) -> str:
    target_characters = set(term.mask_letters.casefold())
    seen = 0
    for index, character in enumerate(token):
        if character.casefold() not in target_characters:
            continue
        if seen == term.mask_occurrence:
            return f"{token[:index]}*{token[index + 1:]}"
        seen += 1
    for index, character in enumerate(token[1:], start=1):
        if character.casefold() in "aeiou\u00e4\u00f6\u00fc":
            return f"{token[:index]}*{token[index + 1:]}"
    return token


def _normalize_form(token: str) -> str:
    normalized = unicodedata.normalize("NFC", token).casefold()
    normalized = (
        normalized.replace("\u00e4", "ae")
        .replace("\u00f6", "oe")
        .replace("\u00fc", "ue")
        .replace("\u00df", "ss")
    )
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", normalized)
        if not unicodedata.combining(character)
    )


_TERM_BY_CANONICAL = {term.canonical: term for term in GERMAN_PROFANITY_TERMS}
_FORM_TO_TERM = {
    _normalize_form(form): term
    for term in GERMAN_PROFANITY_TERMS
    for form in (term.canonical, *term.forms)
}
_ORDERED_PREFIXES = tuple(
    sorted(
        (
            (_normalize_form(prefix), term)
            for term in GERMAN_PROFANITY_TERMS
            for prefix in term.prefixes
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)
