from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .models import Cue, Word
from .text_metrics import token_texts

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_NUMBER_WORDS = {
    "zero": "0",
    "null": "0",
    "one": "1",
    "ein": "1",
    "eine": "1",
    "einen": "1",
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
    "fünf": "5",
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
    "zwölf": "12",
}

_GERMAN_ONES = {
    0: ("null",),
    1: ("eins", "ein", "eine", "einen", "einem", "einer"),
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


def _add_german_numbers() -> None:
    for number, words in {**_GERMAN_ONES, **_GERMAN_TEENS, **_GERMAN_TENS}.items():
        for word in words:
            _NUMBER_WORDS.setdefault(word, str(number))
    for tens in range(20, 100, 10):
        for ones in range(1, 10):
            for one_word in _GERMAN_ONES[ones]:
                for ten_word in _GERMAN_TENS[tens]:
                    _NUMBER_WORDS.setdefault(f"{one_word}und{ten_word}", str(tens + ones))
    for number, words in {
        1: ("erste", "erster", "erstes", "ersten", "erstem"),
        2: ("zweite", "zweiter", "zweites", "zweiten", "zweitem"),
        3: ("dritte", "dritter", "drittes", "dritten", "drittem"),
        4: ("vierte", "vierter", "viertes", "vierten", "viertem"),
        5: ("funfte", "funfter", "funftes", "funften", "funftem", "fuenfte", "fuenfter", "fuenftes", "fuenften", "fuenftem"),
        6: ("sechste", "sechster", "sechstes", "sechsten", "sechstem"),
        7: ("siebte", "siebter", "siebtes", "siebten", "siebtem"),
        8: ("achte", "achter", "achtes", "achten", "achtem"),
        9: ("neunte", "neunter", "neuntes", "neunten", "neuntem"),
        10: ("zehnte", "zehnter", "zehntes", "zehnten", "zehntem"),
    }.items():
        for word in words:
            _NUMBER_WORDS.setdefault(word, str(number))


_add_german_numbers()


@dataclass(frozen=True)
class SRTToken:
    text: str
    normalized: str
    cue_id: int
    token_index: int


def normalize_token(value: str) -> str:
    value = _fold_latin_number_text(unicodedata.normalize("NFC", value).lower())
    parts = TOKEN_RE.findall(value)
    normalized = "".join(_NUMBER_WORDS.get(part, part) for part in parts)
    return _NUMBER_WORDS.get(normalized, normalized)


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
    return "".join(char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char))


def tokenize_cues(cues: list[Cue]) -> list[SRTToken]:
    tokens: list[SRTToken] = []
    for cue in cues:
        for raw in token_texts(cue.plain_text):
            normalized = normalize_token(raw)
            if not normalized:
                continue
            tokens.append(SRTToken(raw, normalized, cue.index, len(tokens)))
    return tokens


def normalized_words(words: list[Word]) -> list[str]:
    return [normalize_token(word.text) for word in words]


def alphanumeric_signature(text: str) -> list[str]:
    return [normalize_token(part) for part in token_texts(text) if normalize_token(part)]
