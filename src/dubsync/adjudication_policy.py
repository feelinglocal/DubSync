"""Conservative wording policies that require no model or invented audio text.

The comparison forms below never become output text. Keeps retain the original
span verbatim; the default spoken register policy can only swap in the performed
ASR register words, in the source's case, keeping every other source character.
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
from .tokenize import alphanumeric_signature, normalize_token, tokenize_cues


DETERMINISTIC_ADJUDICATION_POLICY_VERSION = 2
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
# Japanese has no capitals. A kanji or katakana run followed by an honorific,
# title or place suffix is a name even once (山下様, 天衡グループ, 南山県); a run
# that recurs in two cues and extends such a name is one too (山下森彦). A
# recurring run alone is not: 部下, 屋台 and 今夜 recur as well. Without readings
# only a divergence wholly inside a name keeps the source spelling; a
# same-reading respelling of any other word is still asked.
_JAPANESE_HONORIFICS = (
    "様", "さま", "さん", "君", "くん", "ちゃん", "氏", "殿", "先生", "先輩", "社長", "会長", "グループ",
    "県", "市", "町", "村",
)
_JAPANESE_MIN_NAME_LENGTH = 2
# A respelled name stays name-sized (a kanji reads as a few kana); a longer
# hearing can hold other spoken words.
_JAPANESE_MAX_READING_RATIO = 3
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


def _expansion_steps(
    tokens: tuple[str, ...], forms: dict[tuple[str, ...], tuple[str, ...]],
) -> list[tuple[int, tuple[str, ...]]]:
    """Greedy (consumed token count, expanded tokens) steps behind ``_expand``."""
    steps: list[tuple[int, tuple[str, ...]]] = []
    cursor = 0
    max_length = max(map(len, forms), default=0)
    while cursor < len(tokens):
        for size in range(min(max_length, len(tokens) - cursor), 0, -1):
            replacement = forms.get(tokens[cursor:cursor + size])
            if replacement is not None:
                steps.append((size, replacement))
                cursor += size
                break
        else:
            steps.append((1, tokens[cursor:cursor + 1]))
            cursor += 1
    return steps


def _expand(tokens: tuple[str, ...], forms: dict[tuple[str, ...], tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(token for _, expanded in _expansion_steps(tokens, forms) for token in expanded)


def _expansion_groups(
    source: tuple[str, ...], audio: tuple[str, ...], forms: dict[tuple[str, ...], tuple[str, ...]],
) -> list[tuple[range, range]]:
    """Smallest source/audio token ranges that expand to the same words.

    Callers have already proved that both sides expand to one sequence.
    """
    def boundaries(tokens: tuple[str, ...]) -> dict[int, int]:
        result, consumed, produced = {0: 0}, 0, 0
        for size, expanded in _expansion_steps(tokens, forms):
            consumed, produced = consumed + size, produced + len(expanded)
            result[produced] = consumed
        return result

    source_bounds, audio_bounds = boundaries(source), boundaries(audio)
    cuts = sorted(source_bounds.keys() & audio_bounds.keys())
    return [
        (range(source_bounds[left], source_bounds[right]), range(audio_bounds[left], audio_bounds[right]))
        for left, right in zip(cuts, cuts[1:])
    ]


def _bare_asr_word(text: str, bounds: tuple[int, int]) -> bool:
    """The whitespace-delimited ASR word holds this token plus punctuation only."""
    start, end = bounds
    while start and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return all(unicodedata.category(char).startswith("P") for char in text[start:bounds[0]] + text[bounds[1]:end])


def _source_cased(source: str, words: list[str]) -> list[str]:
    """Spell lower-case register words with the replaced source words' capitals."""
    letters = [char for char in source if char.isalpha()]
    if len(letters) > 1 and all(char.isupper() for char in letters):
        return [word.upper() for word in words]
    if letters and letters[0].isupper():
        return [words[0][:1].upper() + words[0][1:], *words[1:]]
    return words


def _spoken_register_text(span: DivergenceSpan, forms: dict[tuple[str, ...], tuple[str, ...]]) -> str | None:
    """Put the performed register words into the authored span text.

    The audio decides only which register form was spoken. The script keeps
    sentence case and every character outside the replaced words, so ASR
    capitals and attached punctuation never reach the subtitle. An ASR word that
    carries more than its register word, or authored punctuation between the
    replaced words, is left to adjudication.
    """
    source_words, audio_words = token_texts(span.srt_text), token_texts(span.asr_text)
    source_bounds = token_character_spans(span.srt_text, source_words)
    audio_bounds = token_character_spans(span.asr_text, audio_words)
    if source_bounds is None or audio_bounds is None:
        return None
    source, audio = _keys(span.srt_text), _keys(span.asr_text)
    pieces: list[str] = []
    cursor = 0
    for source_range, audio_range in _expansion_groups(source, audio, forms):
        if source[source_range.start:source_range.stop] == audio[audio_range.start:audio_range.stop]:
            continue
        replaced = source_bounds[source_range.start:source_range.stop]
        if any(not span.srt_text[left[1]:right[0]].isspace() for left, right in zip(replaced, replaced[1:])):
            return None
        if not all(_bare_asr_word(span.asr_text, audio_bounds[index]) for index in audio_range):
            return None
        start, end = replaced[0][0], replaced[-1][1]
        words = _source_cased(span.srt_text[start:end], [audio_words[index].lower() for index in audio_range])
        pieces += [span.srt_text[cursor:start], " ".join(words)]
        cursor = end
    return "".join(pieces) + span.srt_text[cursor:]


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


def _japanese_run_script(token: str) -> str | None:
    name = unicodedata.name(token[0], "")
    if token[0] in "々〆" or name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
        return "kanji"
    return "katakana" if "KATAKANA" in name else None


def _japanese_source_names(
    source_cues: Sequence[Cue], language: str | None,
) -> tuple[frozenset[str], tuple[range, ...], tuple[str, ...]]:
    """Japanese names, their source token ranges and the source token keys.

    Tokens are the aligner's ``tokenize_cues`` stream (one per kana or kanji),
    so a divergence span finds itself through its source token indices. Runs
    of one script are split by any other character, including a space.
    """
    if language != "ja":
        return frozenset(), (), ()
    tokens = tokenize_cues(list(source_cues))
    runs: list[tuple[int, tuple[str, ...]]] = []
    cues_by_name: dict[str, set[int]] = defaultdict(set)
    marked: set[str] = set()
    cursor = 0
    for position, cue in enumerate(source_cues):
        text = speech_text_for_alignment(cue)
        count = sum(1 for word in token_texts(text) if normalize_token(word))
        cue_tokens = tokens[cursor:cursor + count]
        cursor += count
        bounds = token_character_spans(text, [token.text for token in cue_tokens])
        if bounds is None:
            continue
        start = 0
        while start < len(cue_tokens):
            # A prolonged-sound or iteration mark continues a word, never starts one.
            script = None if cue_tokens[start].text[0] in "ー々" else _japanese_run_script(cue_tokens[start].text)
            end = start + 1
            while (script is not None and end < len(cue_tokens) and bounds[end - 1][1] == bounds[end][0]
                   and _japanese_run_script(cue_tokens[end].text) == script):
                end += 1
            if script is not None:
                run = tuple(token.text for token in cue_tokens[start:end])
                runs.append((cue_tokens[start].token_index, run))
                # 山下様 names 山下; 貴様, 皆様 or a bare title name nobody.
                title = next((title for title in _JAPANESE_HONORIFICS if "".join(run).endswith(title)), None)
                name = "".join(run[:len(run) - len(title)] if title else run)
                if len(name) >= _JAPANESE_MIN_NAME_LENGTH:
                    cues_by_name[name].add(position)
                    following = unicodedata.normalize("NFKC", text[bounds[end - 1][1]:])
                    if title or following.startswith(_JAPANESE_HONORIFICS):
                        marked.add(name)
            start = end
    names = frozenset(marked | {
        name for name, cue_positions in cues_by_name.items()
        if len(cue_positions) >= 2 and any(name.startswith(known) for known in marked)
    })
    ranges = tuple(
        range(first, first + size)
        for first, run in runs for size in range(1, len(run) + 1) if "".join(run[:size]) in names
    )
    return names, ranges, tuple(token.normalized for token in tokens)


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
        self._japanese_names, self._japanese_name_ranges, self._source_token_keys = _japanese_source_names(
            source_cues or (), self.language,
        )

    def cache_context(self) -> dict[str, object]:
        """Every source-derived policy input, including ambiguous name spellings."""
        return {
            "version": DETERMINISTIC_ADJUDICATION_POLICY_VERSION,
            "language": self.language, "register_policy": self.register_policy,
            "source_names": sorted(self.source_names),
            "name_forms": dict(sorted(self._name_forms.items())),
            "japanese_names": sorted(self._japanese_names),
        }

    def _inside_japanese_name(self, span: DivergenceSpan) -> bool:
        """The span's source tokens all lie in one occurrence of a source name."""
        indices = span.srt_token_indices
        if not self._japanese_name_ranges or not indices:
            return False
        first, last = indices[0], indices[-1]
        if list(indices) != list(range(first, last + 1)) or first < 0 or last >= len(self._source_token_keys):
            return False
        # Indices into another cue list or text cannot borrow this protection.
        if list(self._source_token_keys[first:last + 1]) != alphanumeric_signature(span.srt_text):
            return False
        if len(alphanumeric_signature(span.asr_text)) > _JAPANESE_MAX_READING_RATIO * len(indices):
            return False
        return any(first in occurrence and last in occurrence for occurrence in self._japanese_name_ranges)

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
        if not alphanumeric_signature(span.srt_text) and not alphanumeric_signature(span.asr_text):
            # MAI and Scribe time punctuation (。、？ , . ?) as words. Alone it
            # opens an insertion span in which no letter or digit is spoken.
            return _decision(span, "Punctuation/casing-only difference; preserved source SRT.")
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
        if self._inside_japanese_name(span):
            # The audio cannot choose between homophone spellings of a name.
            return _decision(span, "Recurring source-name spelling equivalent; preserved source SRT.")

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
        spoken_text = _spoken_register_text(span, self._register)
        if spoken_text is None:
            return None
        return _decision(
            span, "Register policy spoken; used the performed register form in source case and punctuation.",
            spoken_text=spoken_text,
        )


def _decision(span: DivergenceSpan, reason: str, *, spoken_text: str | None = None) -> AdjudicationDecision:
    return AdjudicationDecision(
        case_id=span.case_id, verdict="keep_srt" if spoken_text is None else "use_audio",
        final_text=span.srt_text if spoken_text is None else spoken_text, confidence=DETERMINISTIC_KEEP_CONFIDENCE,
        speaker=span.speaker_ids[0] if span.speaker_ids else None, character="unknown", reason=reason,
    )
