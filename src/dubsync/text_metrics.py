from __future__ import annotations

import unicodedata
from collections.abc import Iterable


_CHAR_LEVEL_RANGES = (
    (0x0E00, 0x0E7F),  # Thai
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def contains_character_level_script(text: str) -> bool:
    return any(is_character_level_script(char) for char in text)


def is_character_level_script(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _CHAR_LEVEL_RANGES)


def token_texts(text: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    normalized_text = unicodedata.normalize("NFC", text)
    for index, char in enumerate(normalized_text):
        if is_character_level_script(char) and char.isalnum():
            flush()
            tokens.append(char)
        elif char.isalnum() or char == "_" or _is_inner_hyphen(char, buffer, normalized_text, index):
            buffer.append(char)
        else:
            flush()

    flush()
    return tokens


def _is_inner_hyphen(char: str, buffer: list[str], text: str, index: int) -> bool:
    return char in {"-", "‑"} and bool(buffer) and index + 1 < len(text) and text[index + 1].isalnum()


def wrap_visual_width(text: str, max_width: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    words = stripped.split()
    if len(words) <= 1:
        return _wrap_unspaced(stripped, max_width)
    return _wrap_words(words, max_width)


def _wrap_words(words: Iterable[str], max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if display_width(candidate) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        pieces = _wrap_unspaced(word, max_width)
        if len(pieces) > 1:
            lines.extend(pieces[:-1])
        current = pieces[-1]

    if current:
        lines.append(current)
    return lines


def _wrap_unspaced(text: str, max_width: int) -> list[str]:
    if not text:
        return []
    if display_width(text) <= max_width:
        return [text]
    if not contains_character_level_script(text):
        return _hyphen_split_unspaced(text, max_width)

    lines: list[str] = []
    current = ""
    current_width = 0
    for char in text:
        char_width = display_width(char)
        if current and current_width + char_width > max_width:
            lines.append(current)
            current = char
            current_width = char_width
        else:
            current += char
            current_width += char_width
    if current:
        lines.append(current)
    return lines


def _hyphen_split_unspaced(text: str, max_width: int) -> list[str]:
    if max_width <= 1:
        return [text]

    lines: list[str] = []
    current = ""
    current_width = 0
    hyphen_width = display_width("-")
    split_width = max_width - hyphen_width

    for char in text:
        char_width = display_width(char)
        if current and current_width + char_width > split_width:
            lines.append(f"{current}-")
            current = char
            current_width = char_width
        else:
            current += char
            current_width += char_width

    if current:
        lines.append(current)
    return lines
