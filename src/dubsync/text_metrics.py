from __future__ import annotations

import unicodedata
from collections.abc import Iterable


_UNSPACED_RANGES = (
    (0x3005, 0x3007),  # Ideographic iteration mark, closing mark, and zero
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # Half-width Katakana, including voicing marks
    (0x1AFF0, 0x1AFFF),  # Kana Extended-B
    (0x1B000, 0x1B16F),  # Kana Supplement, Extended-A, and Small Kana Extension
    (0x20000, 0x2EE5F),  # Supplementary CJK unified ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x30000, 0x323AF),  # CJK Extensions G and H
)
_CHAR_LEVEL_RANGES = _UNSPACED_RANGES + (
    (0x0E00, 0x0E7F),  # Thai
    (0xAC00, 0xD7AF),  # Hangul syllables
)

# Best-effort kinsoku: prefer a legal break without discarding characters or
# blocking output when a very narrow requested width cannot fit the cluster.
_OPENING_PUNCTUATION = frozenset("([{‘“〈《「『【〔〖〘〚（［｛｟")
_CLOSING_PUNCTUATION = frozenset(")]}’”〉》」』】〕〗〙〛）］｝｠、。，．・：；？！,.!?:;%")
_NONSTARTING_KANA = frozenset("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶー々ゝゞヽヾ")
_JAPANESE_PUNCTUATION = frozenset("〈《「『【〔〖〘〚（［｛｟〉》」』】〕〗〙〛）］｝｠、。，．・：；？！")


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char) or unicodedata.category(char) in {"Mn", "Me"}:
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

    # Comparison tokens canonicalize width variants; callers retain source text.
    normalized_text = unicodedata.normalize("NFKC", text)
    for index, char in enumerate(normalized_text):
        if is_character_level_script(char) and char.isalnum():
            flush()
            tokens.append(char)
        elif (
            char in {"\u3099", "\u309a"}
            and tokens
            and index > 0
            and normalized_text[index - 1].isalnum()
            and _is_unspaced_script(normalized_text[index - 1])
        ):
            # Some kana + voicing combinations have no precomposed codepoint.
            tokens[-1] += char
        elif char.isalnum() or char == "_" or _is_inner_hyphen(char, buffer, normalized_text, index):
            buffer.append(char)
        else:
            flush()

    flush()
    return tokens


def join_word_texts(texts: Iterable[str]) -> str:
    """Join ASR/line fragments without inserting spaces into Japanese text."""
    result: list[str] = []
    previous = ""
    for text in texts:
        current = text.strip()
        if not current:
            continue
        if previous and not _joins_without_space(previous, current):
            result.append(" ")
        result.append(current)
        previous = current
    return "".join(result)


def _joins_without_space(previous: str, current: str) -> bool:
    left = unicodedata.normalize("NFKC", previous[-1])[-1]
    right = unicodedata.normalize("NFKC", current[0])[0]
    return (
        _is_unspaced_script(left)
        or _is_unspaced_script(right)
        or left in _OPENING_PUNCTUATION
        or right in _CLOSING_PUNCTUATION
        or left in _JAPANESE_PUNCTUATION
        or right in _JAPANESE_PUNCTUATION
        or bool(unicodedata.combining(right))
    )


def _is_unspaced_script(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _UNSPACED_RANGES)


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

    clusters = _text_clusters(text)
    widths = [display_width(cluster) for cluster in clusters]
    lines: list[str] = []
    start = 0
    while start < len(clusters):
        end = start
        width = 0
        while end < len(clusters) and (end == start or width + widths[end] <= max_width):
            width += widths[end]
            end += 1
        if end < len(clusters):
            fitted_end = end
            while end > start and not _can_break_between(clusters[end - 1], clusters[end]):
                end -= 1
            if end == start:
                # No legal break fits. Keep punctuation/voicing attached, even
                # if the resulting line exceeds the advisory width preference.
                end = fitted_end
                while end < len(clusters) and not _can_break_between(clusters[end - 1], clusters[end]):
                    end += 1
        lines.append("".join(clusters[start:end]))
        start = end
    return lines


def _text_clusters(text: str) -> list[str]:
    clusters: list[str] = []
    for char in text:
        is_mark = bool(unicodedata.combining(char)) or unicodedata.category(char) in {"Mn", "Me"}
        if clusters and (is_mark or char in {"\uff9e", "\uff9f"}):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _can_break_between(left: str, right: str) -> bool:
    left_char = unicodedata.normalize("NFKC", left)[-1]
    right_char = unicodedata.normalize("NFKC", right)[0]
    return (
        left_char not in _OPENING_PUNCTUATION
        and right_char not in _CLOSING_PUNCTUATION
        and right_char not in _NONSTARTING_KANA
        and not 0x31F0 <= ord(right_char) <= 0x31FF
    )


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
