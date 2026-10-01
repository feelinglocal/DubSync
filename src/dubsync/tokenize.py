from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from .models import Cue, Word
from .profanity import normalize_german_profanity_token
from .subtitle_annotations import speech_text_for_alignment
from .text_metrics import token_texts

TOKEN_RE = re.compile(r"[\w\u3099\u309a]+", re.UNICODE)
_STUTTER_SPLIT_RE = re.compile("[-\u2011]")
# ``1.000`` / ``1,000`` group thousands; ``2,5`` / ``2.5`` are one decimal.
_GROUPED_NUMBER_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+")
_DECIMAL_NUMBER_RE = re.compile(r"\d+[.,]\d+")
_NUMBER_EDGE_PUNCTUATION = " .,;:!?\u2026\"'()[]\u00ab\u00bb\u201c\u201d\u201e\u2018\u2019"
_NUMBER_WORDS = {
    "zero": "0",
    "null": "0",
    "one": "1",
    "eins": "1",
    "two": "2",
    "zwei": "2",
    "three": "3",
    "drei": "3",
    "four": "4",
    "vier": "4",
    "five": "5",
    "fuenf": "5",
    "funf": "5",
    "f\u00fcnf": "5",
    "six": "6",
    "sechs": "6",
    "seven": "7",
    "sieben": "7",
    "eight": "8",
    "acht": "8",
    "nine": "9",
    "neun": "9",
    "ten": "10",
    "zehn": "10",
    "eleven": "11",
    "elf": "11",
    "twelve": "12",
    "zwoelf": "12",
    "zwolf": "12",
    "zw\u00f6lf": "12",
    "%": "prozent",
    "percent": "prozent",
    "porcento": "prozent",
    "porciento": "prozent",
    "pourcent": "prozent",
}

_GERMAN_ONES = {
    0: ("null",),
    1: ("eins",),
    2: ("zwei", "zwo"),
    3: ("drei",),
    4: ("vier",),
    5: ("fuenf", "funf"),
    6: ("sechs",),
    7: ("sieben",),
    8: ("acht",),
    9: ("neun",),
}
_GERMAN_TEENS = {
    10: ("zehn",),
    11: ("elf",),
    12: ("zwoelf", "zwolf"),
    13: ("dreizehn",),
    14: ("vierzehn",),
    15: ("fuenfzehn", "funfzehn"),
    16: ("sechzehn",),
    17: ("siebzehn",),
    18: ("achtzehn",),
    19: ("neunzehn",),
}
_GERMAN_TENS = {
    20: ("zwanzig",),
    30: ("dreissig", "dreizig"),
    40: ("vierzig",),
    50: ("fuenfzig", "funfzig"),
    60: ("sechzig",),
    70: ("siebzig",),
    80: ("achtzig",),
    90: ("neunzig",),
    100: ("hundert", "einhundert"),
}


# Words that are also articles, ordinals or ordinary words in a supported
# language ("ein", "um", "erste", French "neuf" = new, Portuguese "dos" = of
# the). They never share a key with each other or with the number; they only
# match a token written in digits (``eine`` / ``1``).
NUMBER_ALIASES: dict[str, str] = {}
_ARTICLE_KEYS: dict[str, str] = {}
# Joins a spoken multi-word number ("vinte e seis", "twenty and six").
NUMBER_CONNECTORS = frozenset({"e", "y", "et", "und", "and"})
# Spoken two-word percent suffixes, as comparison keys ("por cento" -> por 100).
_PERCENT_SUFFIXES = frozenset({("por", "100"), ("pour", "cent"), ("per", "cent")})
# Accent marks that change the word and its sound; every other Latin accent is
# folded because the two spellings are orthographic variants of one word.
# Portuguese "é"/"e" stays folded: adjudication_regions bridges that collision.
_ACCENT_SIGNIFICANT = frozenset({
    "está", "estás", "esté", "avó", "avô", "dê", "pôr", "pôde",
})
# Accented words whose folded spelling is a number word in another language.
_ACCENTED_NON_NUMBERS = frozenset({"très"})


def _add_german_numbers() -> None:
    for number, words in {**_GERMAN_ONES, **_GERMAN_TEENS, **_GERMAN_TENS}.items():
        for word in words:
            _NUMBER_WORDS.setdefault(word, str(number))
    rest = _german_numbers_below_hundred()
    for value, words in rest.items():
        for word in words:
            _NUMBER_WORDS.setdefault(word, str(value))
    # hundertfünfzig, zweihundertdrei, neunzehnhundertneunundneunzig, zweitausendzwanzig
    hundreds = {1: ("", "ein"), **{ones: _GERMAN_ONES[ones] for ones in range(2, 10)}, **_GERMAN_TEENS}
    for multiplier, prefixes in hundreds.items():
        for prefix in prefixes:
            if multiplier == 10:
                continue
            base = f"{prefix}hundert"
            _NUMBER_WORDS.setdefault(base, str(multiplier * 100))
            for value, words in rest.items():
                for word in words:
                    _NUMBER_WORDS.setdefault(f"{base}{word}", str(multiplier * 100 + value))
                    _NUMBER_WORDS.setdefault(f"{base}und{word}", str(multiplier * 100 + value))
    for prefix, multiplier in (("", 1), ("ein", 1), ("zwei", 2)):
        base = f"{prefix}tausend"
        _NUMBER_WORDS.setdefault(base, str(multiplier * 1000))
        for value, words in rest.items():
            for word in words:
                _NUMBER_WORDS.setdefault(f"{base}{word}", str(multiplier * 1000 + value))
                _NUMBER_WORDS.setdefault(f"{base}und{word}", str(multiplier * 1000 + value))
    # The indefinite article keeps one key for its inflections (ending errors
    # are common ASR noise) but no longer equals "eins", "1" or "erste".
    for article in ("eine", "einen", "einem", "einer", "eines"):
        _ARTICLE_KEYS[article] = "ein"
    NUMBER_ALIASES["ein"] = "1"
    for number, stem in enumerate(
        ("erst", "zweit", "dritt", "viert", "fuenft", "sechst", "siebt", "acht", "neunt", "zehnt"), start=1,
    ):
        for ending in ("e", "er", "es", "en", "em"):
            NUMBER_ALIASES[f"{stem}{ending}"] = str(number)
            if stem == "fuenft":
                NUMBER_ALIASES[f"funft{ending}"] = str(number)


def _german_numbers_below_hundred() -> dict[int, tuple[str, ...]]:
    rest: dict[int, tuple[str, ...]] = {1: ("eins",)}
    for ones in range(2, 10):
        rest[ones] = _GERMAN_ONES[ones]
    rest.update(_GERMAN_TEENS)
    for tens in range(20, 100, 10):
        rest[tens] = _GERMAN_TENS[tens]
        for ones in range(1, 10):
            rest[tens + ones] = tuple(
                f"{one_word}und{ten_word}"
                for one_word in (("ein",) if ones == 1 else _GERMAN_ONES[ones])
                for ten_word in _GERMAN_TENS[tens]
            )
    return rest


def _add_other_numbers() -> None:
    """Portuguese, Spanish, French and English cardinals (accent-folded keys)."""

    cardinals = {
        # Portuguese
        "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9,
        "dez": 10, "onze": 11, "doze": 12, "treze": 13, "catorze": 14, "quatorze": 14, "quinze": 15,
        "dezesseis": 16, "dezasseis": 16, "dezessete": 17, "dezassete": 17, "dezoito": 18, "dezenove": 19,
        "dezanove": 19, "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60,
        "setenta": 70, "oitenta": 80, "noventa": 90, "cem": 100, "cento": 100, "mil": 1000,
        # Spanish
        "cuatro": 4, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "trece": 13, "catorce": 14, "quince": 15,
        "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "treinta": 30,
        "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "ochenta": 80, "cien": 100, "ciento": 100,
        # French
        "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "sept": 7, "huit": 8, "dix": 10, "douze": 12,
        "treize": 13, "dixsept": 17, "dixhuit": 18, "dixneuf": 19, "vingt": 20, "trente": 30, "quarante": 40,
        "cinquante": 50, "soixante": 60, "mille": 1000,
        # English
        "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    }
    for prefix, value in (("duz", 2), ("trez", 3), ("quatroc", 4), ("quinh", 5), ("seisc", 6), ("setec", 7),
                          ("oitoc", 8), ("novec", 9)):
        cardinals[f"{prefix}entos"] = value * 100
        cardinals[f"{prefix}entas"] = value * 100
    for prefix, value in (("dosc", 2), ("tresc", 3), ("cuatroc", 4), ("seisc", 6), ("setec", 7), ("ochoc", 8),
                          ("novec", 9)):
        cardinals[f"{prefix}ientos"] = value * 100
        cardinals[f"{prefix}ientas"] = value * 100
    cardinals.update({"quinientos": 500, "quinientas": 500})
    spanish_units = {"uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
                     "siete": 7, "ocho": 8, "nueve": 9}
    for word, value in spanish_units.items():
        cardinals[f"veinti{word}"] = 20 + value
    french_units = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9}
    for tens_word, tens in (("vingt", 20), ("trente", 30), ("quarante", 40), ("cinquante", 50), ("soixante", 60)):
        cardinals[f"{tens_word}etun"] = tens + 1
        for word, value in french_units.items():
            cardinals[f"{tens_word}{word}"] = tens + value
    french_teens = {"dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
                    "dixsept": 17, "dixhuit": 18, "dixneuf": 19}
    for word, value in french_teens.items():
        cardinals[f"soixante{word}"] = 60 + value
        cardinals[f"quatrevingt{word}"] = 80 + value
    cardinals["soixanteetonze"] = 71
    cardinals.update({"quatrevingts": 80, "quatrevingt": 80, "quatrevingtun": 81})
    for word, value in french_units.items():
        cardinals[f"quatrevingt{word}"] = 80 + value
    english_units = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                     "nine": 9}
    for tens_word, tens in (("twenty", 20), ("thirty", 30), ("forty", 40), ("fifty", 50), ("sixty", 60),
                            ("seventy", 70), ("eighty", 80), ("ninety", 90)):
        for word, value in english_units.items():
            cardinals[f"{tens_word}{word}"] = tens + value
    for word, value in cardinals.items():
        _NUMBER_WORDS.setdefault(word, str(value))
    NUMBER_ALIASES.update({
        # articles
        "um": "1", "uma": "1", "un": "1", "une": "1", "uno": "1", "una": "1",
        # ordinary words elsewhere: pt "dos" (of the) / "doce" (sweet), en "once", fr "neuf" (new),
        # en "seize", en "cent"
        "dos": "2", "doce": "12", "once": "11", "neuf": "9", "seize": "16", "cent": "100",
    })


_add_german_numbers()
_add_other_numbers()


def number_value(key: str) -> int | None:
    """The cardinal a comparison key spells, including article-like aliases."""
    if key.isdigit():
        return int(key)
    alias = NUMBER_ALIASES.get(key)
    return int(alias) if alias is not None else None


def spoken_number_values(keys: list[str]) -> frozenset[int]:
    """Values a sequence of spoken number keys can denote (``20 e 6`` -> 26).

    Additive readings must name strictly smaller parts after each multiplier
    (``mil quinhentos e vinte``); a run of single digits is also read digit by
    digit (``um nove zero`` -> 190, a flight or room number).
    """
    values: list[int] = []
    connectors = 0
    for position, key in enumerate(keys):
        if key in NUMBER_CONNECTORS:
            if position in {0, len(keys) - 1}:
                return frozenset()
            connectors += 1
            continue
        value = number_value(key)
        if value is None:
            return frozenset()
        values.append(value)
    if not values:
        return frozenset()
    readings: set[int] = set()
    if len(values) >= 2 and not connectors and all(value <= 9 for value in values):
        readings.add(int("".join(str(value) for value in values)))
    additive = _additive_number(values)
    if additive is not None:
        readings.add(additive)
    return frozenset(readings)


def _additive_number(values: list[int]) -> int | None:
    total = 0
    current = 0
    previous: int | None = None
    for value in values:
        if value == 1000:
            total += (current or 1) * 1000
            current = 0
            previous = None
            continue
        if value == 100 and 0 < current < 10:
            current *= 100
            previous = None
            continue
        if previous is not None and value >= previous:
            return None
        current += value
        previous = value
    return total + current


def percent_suffix_length(keys: list[str]) -> int:
    """How many trailing keys spell "percent" (``%``, ``Prozent``, ``por cento``)."""
    if keys and keys[-1] == "prozent":
        return 1
    if len(keys) >= 2 and (keys[-2], keys[-1]) in _PERCENT_SUFFIXES:
        return 2
    return 0


@dataclass(frozen=True)
class SRTToken:
    text: str
    normalized: str
    cue_id: int
    token_index: int


@lru_cache(maxsize=16_384)
def normalize_token(value: str) -> str:
    profanity = normalize_german_profanity_token(value)
    if profanity is not None:
        return profanity
    lowered = unicodedata.normalize("NFKC", value).lower()
    accented = unicodedata.normalize("NFC", "".join(TOKEN_RE.findall(lowered)))
    if accented in _ACCENT_SIGNIFICANT:
        # Portuguese "é" (is) and "e" (and) sound different and are different words.
        return accented
    value = _without_glued_stutter(_fold_latin_number_text(lowered))
    if accented in _ACCENTED_NON_NUMBERS:
        return "".join(TOKEN_RE.findall(value))
    if value in _NUMBER_WORDS:
        return _NUMBER_WORDS[value]
    number = value.strip(_NUMBER_EDGE_PUNCTUATION)
    if _GROUPED_NUMBER_RE.fullmatch(number):
        return number.replace(".", "").replace(",", "")
    if _DECIMAL_NUMBER_RE.fullmatch(number):
        return number.replace(",", ".")
    parts = TOKEN_RE.findall(value)
    joined = "".join(parts)
    if joined in _NUMBER_WORDS:
        # Hyphenated number words: dix-sept, vingt-deux, twenty-one.
        return _NUMBER_WORDS[joined]
    if joined in _ARTICLE_KEYS:
        return _ARTICLE_KEYS[joined]
    normalized = "".join(_NUMBER_WORDS.get(part, part) for part in parts)
    return _NUMBER_WORDS.get(normalized, normalized)


def _without_glued_stutter(value: str) -> str:
    """Key a glued stutter (``N-não``, ``De-deixa``, ``eu-eu``) like its word.

    Only hyphen-joined fragments that repeat the start of the final word (up to
    three letters) or the whole word count; ordinary compounds keep every part.
    """
    segments = _STUTTER_SPLIT_RE.split(value)
    if len(segments) < 2:
        return value
    keys = ["".join(TOKEN_RE.findall(segment)) for segment in segments]
    word = keys[-1]
    if not word.isalpha() or not all(
        key and (key == word or (len(key) <= 3 and len(key) < len(word) and word.startswith(key)))
        for key in keys[:-1]
    ):
        return value
    return segments[-1]


def _fold_latin_number_text(value: str) -> str:
    value = (
        value.replace("\u00c3\u00a4", "ae")
        .replace("\u00c3\u00b6", "oe")
        .replace("\u00c3\u00bc", "ue")
        .replace("\u00c3\u009f", "ss")
    )
    value = value.translate(
        str.maketrans(
            {
                "\u00e4": "ae",
                "\u00f6": "oe",
                "\u00fc": "ue",
                "\u00df": "ss",
            }
        )
    )
    folded: list[str] = []
    latin_base = False
    for char in unicodedata.normalize("NFKD", value):
        if unicodedata.combining(char):
            # Accents can be folded for Latin matching, but kana voicing is lexical.
            if not latin_base:
                folded.append(char)
        else:
            latin_base = "LATIN" in unicodedata.name(char, "")
            folded.append(char)
    return unicodedata.normalize("NFC", "".join(folded))


def tokenize_cues(cues: list[Cue]) -> list[SRTToken]:
    tokens: list[SRTToken] = []
    for cue in cues:
        for raw in token_texts(speech_text_for_alignment(cue)):
            normalized = normalize_token(raw)
            if not normalized:
                continue
            tokens.append(SRTToken(raw, normalized, cue.index, len(tokens)))
    return tokens


def normalized_words(words: list[Word]) -> list[str]:
    return [normalize_token(word.text) for word in words]


def alphanumeric_signature(text: str) -> list[str]:
    return [normalized for part in token_texts(text) if (normalized := normalize_token(part))]
