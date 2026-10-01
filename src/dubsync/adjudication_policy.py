"""Conservative wording policies that require no model or invented audio text.

The comparison forms below never become output text. Keeps retain the original
span verbatim; the default spoken register policy can only return exact ASR text.
Unknown languages get the existing punctuation/casing comparison only.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence

from .models import AdjudicationDecision, Cue, DivergenceSpan
from .subtitle_annotations import speech_text_for_alignment
from .text_metrics import markup_spans, token_character_spans, token_texts
from .tokenize import alphanumeric_signature, normalize_token


DETERMINISTIC_ADJUDICATION_POLICY_VERSION = 1
DETERMINISTIC_KEEP_CONFIDENCE = 1.0

# Deliberately finite, language-scoped spellings. Compact-string equality alone
# confuses "a part/apart", "por que/porque", "re-sign/resign" and digit runs.
_SPACING_FORMS = {
    "pt": (
        ("ano-novo", "ano novo"),
        *((f"bem-sucedid{ending}", f"bem sucedid{ending}") for ending in ("o", "a", "os", "as")),
        *((f"superpreocupad{ending}", f"super preocupad{ending}") for ending in ("o", "a", "os", "as")),
    ),
    "de": (("e-mail", "email"), ("habs", "hab s"), ("wärs", "wär s"), ("gibts", "gibt s")),
    "en": (("e-mail", "email"),),
}
_ABBREVIATIONS = {
    "pt": (("sr", "senhor"), ("sra", "senhora"), ("srta", "senhorita"), ("dr", "doutor"), ("dra", "doutora")),
    "de": (("hr", "herr"), ("fr", "frau"), ("dr", "doktor")),
    "en": (("mr", "mister"), ("mrs", "missus"), ("dr", "doctor")),
}
_REGISTER_FORMS = {
    "pt": (
        ("pra", "para"), ("pro", "para o"), ("pras", "para as"), ("pros", "para os"),
        ("tá", "está"), ("tô", "estou"), ("tava", "estava"), ("tavam", "estavam"),
        ("cê", "você"), ("vamo", "vamos"),
    ),
    "de": (("hab", "habe"), ("nich", "nicht")),
    "en": (("gonna", "going to"), ("wanna", "want to")),
}
_NAME_LANGUAGES = frozenset({"pt", "en", "de", "fr", "es"})
_NAME_TITLES = frozenset({
    "sr", "sra", "srta", "senhor", "senhora", "senhorita", "dr", "dra", "doutor", "doutora",
    "hr", "herr", "herrn", "fr", "frau", "doktor", "mr", "mrs", "miss", "mister", "doctor",
})
# These common capitalised fragments are not evidence of a personal name.
_NAME_STOPWORDS = frozenset({
    "a", "o", "as", "os", "um", "uma", "e", "eu", "ele", "ela", "eles", "elas", "você", "vocês",
    "isso", "isto", "esse", "essa", "sim", "não", "nao", "mas", "para", "por", "com", "que", "quem",
    "quando", "onde", "como", "além", "alem", "obrigado", "de", "do", "da", "dos", "das",
    "i", "you", "he", "she", "it", "we", "they", "the", "a", "an", "and", "but", "no", "yes",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "der", "die", "das", "ein", "eine", "und",
}) | _NAME_TITLES
_SENTENCE_END_RE = re.compile(r"[.!?…。！？]")
_HYPHEN_RE = re.compile("[-\u2010\u2011]")


def _keys(text: str) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", token).casefold().replace("\u2011", "-") for token in token_texts(text))


def _language_key(language: str | None) -> str | None:
    return language.strip().lower().replace("_", "-").split("-", 1)[0] if language else None


def _glued_stutter(token: str) -> bool:
    parts = _HYPHEN_RE.split(token)
    word = parts[-1]
    return len(parts) > 1 and word.isalpha() and all(
        part == word or (0 < len(part) <= 3 and len(part) < len(word) and word.startswith(part))
        for part in parts[:-1]
    )


def _form_map(forms: Sequence[tuple[str, str]]) -> dict[tuple[str, ...], tuple[str, ...]]:
    return {_keys(short): _keys(full) for short, full in forms}


def _expand(tokens: tuple[str, ...], forms: dict[tuple[str, ...], tuple[str, ...]]) -> tuple[str, ...]:
    result: list[str] = []
    cursor = 0
    max_length = max(map(len, forms), default=0)
    while cursor < len(tokens):
        for size in range(min(max_length, len(tokens) - cursor), 0, -1):
            replacement = forms.get(tokens[cursor:cursor + size])
            if replacement is not None:
                result.extend(replacement)
                cursor += size
                break
        else:
            result.append(tokens[cursor])
            cursor += 1
    return tuple(result)


def _name_key(tokens: tuple[str, ...]) -> str:
    """Only narrow spelling alternations, never unrestricted fuzzy similarity.

    Recurring names such as Dony/Doni/Donny/Donnie and Luan Nian/Luanian share
    these forms. Vowel changes, deleted syllables and other names remain review
    cases even when edit distance is small.
    """
    compact = "".join(tokens).replace("y", "i")
    if compact.endswith("ie"):
        compact = compact[:-1]
    return re.sub(r"([^aeiou])\1+", r"\1", compact)


def _source_name_lexicon(
    source_cues: Sequence[Cue], language: str | None,
) -> tuple[frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]]:
    if language not in _NAME_LANGUAGES:
        return frozenset(), frozenset()
    occurrences: Counter[tuple[str, ...]] = Counter()
    capitalised: Counter[tuple[str, ...]] = Counter()
    distinct_cues: dict[tuple[str, ...], set[int]] = defaultdict(set)
    observed: set[tuple[str, ...]] = set()
    for cue in source_cues:
        text = speech_text_for_alignment(cue)
        words = token_texts(text)
        bounds = token_character_spans(text, words)
        if bounds is None:
            continue
        keys = tuple(word.casefold() for word in words)
        for position, word in enumerate(words):
            previous = keys[position - 1] if position else None
            titled = previous in _NAME_TITLES
            sentence_initial = not position or (
                not titled and _SENTENCE_END_RE.search(text[bounds[position - 1][1]:bounds[position][0]]) is not None
            )
            if sentence_initial or (language == "de" and not titled):
                continue
            for size in range(1, min(3, len(words) - position) + 1):
                phrase = keys[position:position + size]
                raw_phrase = words[position:position + size]
                if any(
                    not part.isalpha() or len(part) < 3 or part in _NAME_STOPWORDS or normalize_token(part).isdigit()
                    for part in phrase
                ):
                    break
                if size > 1 and any(
                    _SENTENCE_END_RE.search(text[bounds[index - 1][1]:bounds[index][0]])
                    for index in range(position + 1, position + size)
                ):
                    break
                occurrences[phrase] += 1
                if all(part.istitle() for part in raw_phrase):
                    capitalised[phrase] += 1
                    distinct_cues[phrase].add(cue.index)
                    observed.add(phrase)
    recurring = frozenset(
        phrase for phrase, count in capitalised.items()
        if len(distinct_cues[phrase]) >= 2 and count / occurrences[phrase] >= 0.8 and len("".join(phrase)) >= 4
    )
    return recurring, frozenset(observed)


def build_source_name_lexicon(
    source_cues: Sequence[Cue], language: str | None = None,
) -> frozenset[tuple[str, ...]]:
    """Recurring source names as casefolded token tuples, for prompt/risk use."""
    return _source_name_lexicon(source_cues, _language_key(language))[0]


class DeterministicAdjudicationPolicy:
    def __init__(
        self, source_cues: Sequence[Cue] | None = None, language: str | None = None,
        register_policy: str = "spoken",
    ):
        if register_policy not in ("script", "spoken"):
            raise ValueError("adjudication.register_policy must be script or spoken")
        self.language = _language_key(language)
        self.register_policy = register_policy
        self.source_names, observed = _source_name_lexicon(source_cues or (), self.language)
        spellings_by_key: dict[str, set[str]] = defaultdict(set)
        for phrase in observed:
            spellings_by_key[_name_key(phrase)].add("".join(phrase))
        # Two different names actually authored in the episode make the match
        # ambiguous, even if only one spelling recurs enough to be in the lexicon.
        self._name_forms = {
            _name_key(phrase): "@name:" + "".join(phrase)
            for phrase in self.source_names
            if len(spellings_by_key[_name_key(phrase)]) == 1
        }
        self._abbreviations = _form_map(_ABBREVIATIONS.get(self.language, ()))
        self._spacing = _form_map(_SPACING_FORMS.get(self.language, ()))
        self._register = _form_map(_REGISTER_FORMS.get(self.language, ()))

    def cache_context(self) -> dict[str, object]:
        """Every source-derived policy input, including ambiguous name spellings."""
        return {
            "version": DETERMINISTIC_ADJUDICATION_POLICY_VERSION,
            "language": self.language, "register_policy": self.register_policy,
            "source_names": sorted(self.source_names),
            "name_forms": dict(sorted(self._name_forms.items())),
        }

    def _names(self, tokens: tuple[str, ...], *, source: bool) -> tuple[str, ...]:
        result: list[str] = []
        cursor = 0
        while cursor < len(tokens):
            for size in range(min(3, len(tokens) - cursor), 0, -1):
                phrase = tokens[cursor:cursor + size]
                if source and phrase not in self.source_names:
                    continue
                if any(not part.isalpha() or len(part) < 3 or part in _NAME_STOPWORDS for part in phrase):
                    continue
                replacement = self._name_forms.get(_name_key(phrase))
                if replacement is not None:
                    result.append(replacement)
                    cursor += size
                    break
            else:
                result.append(tokens[cursor])
                cursor += 1
        return tuple(result)

    def decide(self, span: DivergenceSpan) -> AdjudicationDecision | None:
        source, audio = _keys(span.srt_text), _keys(span.asr_text)
        if not source or not audio:
            return None
        if source == audio:
            return _decision(span, "Punctuation/casing-only difference; preserved source SRT.")
        # Preserve existing numeric/accent/stutter comparisons, but do not let
        # normalize_token erase semantic inner hyphens such as re-sign/resign.
        hyphens_differ = (
            any(_HYPHEN_RE.search(token) and not _glued_stutter(token) for token in (*source, *audio))
            and source != audio
        )
        if not hyphens_differ and alphanumeric_signature(span.srt_text) == alphanumeric_signature(span.asr_text):
            return _decision(span, "Punctuation/casing-only difference; preserved source SRT.")
        if not self.language:
            return None

        source_spelling = _expand(_expand(source, self._spacing), self._abbreviations)
        audio_spelling = _expand(_expand(audio, self._spacing), self._abbreviations)
        source_named = self._names(source_spelling, source=True)
        audio_named = self._names(audio_spelling, source=False)
        if source_named == audio_named:
            if source_spelling != audio_spelling:
                reason = "Recurring source-name spelling equivalent; preserved source SRT."
            elif _expand(source, self._spacing) == _expand(audio, self._spacing):
                reason = "Equivalent spacing/hyphenation; preserved source SRT."
            else:
                reason = "Language-scoped abbreviation equivalent; preserved source SRT."
            return _decision(span, reason)

        if _expand(source_named, self._register) != _expand(audio_named, self._register):
            return None
        if self.register_policy == "script":
            return _decision(span, "Register policy script; preserved source SRT.")
        # Spoken output is allowed only when the *entire* literal ASR differs by
        # register. It may not also expand an abbreviation, respell a name, or
        # discard authored markup/line breaks. Those cases still need review.
        if (
            _expand(source, self._register) != _expand(audio, self._register)
            or markup_spans(span.srt_text) or "\n" in span.srt_text or "\r" in span.srt_text
        ):
            return None
        return _decision(span, "Register policy spoken; used exact ASR wording.", spoken=True)


def _decision(span: DivergenceSpan, reason: str, *, spoken: bool = False) -> AdjudicationDecision:
    return AdjudicationDecision(
        case_id=span.case_id, verdict="use_audio" if spoken else "keep_srt",
        final_text=span.asr_text if spoken else span.srt_text, confidence=DETERMINISTIC_KEEP_CONFIDENCE,
        speaker=span.speaker_ids[0] if span.speaker_ids else None, character="unknown", reason=reason,
    )
