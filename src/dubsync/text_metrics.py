from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

# SRT/WebVTT inline tags and ASS/SSA override blocks. Only known tag names are
# hidden, so dialogue such as "a < b" keeps every word.
_MARKUP_RE = re.compile(
    r"</?(?:i|b|u|s|em|strong|font|c|v|lang|ruby|rt|span)(?:[\s.][^<>]*)?>"
    r"|<\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?>"
    r"|\{\\[^{}]*\}",
    re.IGNORECASE,
)


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
    hidden = _markup_mask(text)
    for index, char in enumerate(text):
        # Inline tags are not displayed.
        if hidden is not None and hidden[index]:
            continue
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
    hidden = _markup_mask(normalized_text)
    for index, char in enumerate(normalized_text):
        if hidden is not None and hidden[index]:
            # Inline tags are layout, never speech.
            flush()
            continue
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
        elif char == "%" and index > 0 and normalized_text[index - 1].isdigit():
            flush()
            tokens.append(char)
        elif (
            char.isalnum()
            or char == "_"
            or _is_inner_mask(char, buffer, normalized_text, index)
            or _is_inner_hyphen(char, buffer, normalized_text, index)
            or _is_inner_number_separator(char, buffer, normalized_text, index)
        ):
            buffer.append(char)
        else:
            flush()

    flush()
    return tokens


def markup_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of inline subtitle tags (``<i>``, ``<font ...>``, ``{\\an8}``)."""
    return [match.span() for match in _MARKUP_RE.finditer(text)] if ("<" in text or "{" in text) else []


def _markup_mask(text: str) -> list[bool] | None:
    spans = markup_spans(text)
    if not spans:
        return None
    mask = [False] * len(text)
    for start, end in spans:
        mask[start:end] = [True] * (end - start)
    return mask


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
    return char in {"-", "\u2011"} and bool(buffer) and index + 1 < len(text) and text[index + 1].isalnum()


def _is_inner_number_separator(char: str, buffer: list[str], text: str, index: int) -> bool:
    """Keep ``1.000`` and ``2,5`` whole, like a speaker says them."""
    return (
        char in {".", ","}
        and bool(buffer)
        and buffer[-1].isdigit()
        and index + 1 < len(text)
        and text[index + 1].isdigit()
        and all(part.isdigit() or part in {".", ","} for part in buffer)
    )


def _is_inner_mask(char: str, buffer: list[str], text: str, index: int) -> bool:
    if char != "*" or not buffer:
        return False
    next_index = index + 1
    while next_index < len(text) and text[next_index] == "*":
        next_index += 1
    return next_index < len(text) and text[next_index].isalnum()


def wrap_visual_width(text: str, max_width: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    words = stripped.split()
    if len(words) <= 1:
        return _wrap_unspaced(stripped, max_width)
    greedy = _wrap_words(words, max_width)
    if len(greedy) <= 1 or len(words) > 128 or len(greedy) > 8:
        return greedy
    widths = [display_width(word) for word in words]
    # Keep the established CJK/long-word splitter when a word cannot fit.
    if any(width > max_width for width in widths) or markup_spans(stripped):
        return greedy
    return _balanced_word_lines(words, widths, max_width, len(greedy))


def _balanced_word_lines(words: list[str], widths: list[int], max_width: int, line_count: int) -> list[str]:
    """Balance the few subtitle lines without increasing their minimum count."""
    prefix = [0]
    for width in widths:
        prefix.append(prefix[-1] + width)
    target = (prefix[-1] + len(words) - line_count) / line_count
    # (number of lines, consumed words) -> (cost, line-end positions).
    previous: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for line_number in range(line_count):
        current: dict[int, tuple[float, tuple[int, ...]]] = {}
        for start, (cost, breaks) in previous.items():
            for end in range(start + 1, len(words) + 1):
                width = prefix[end] - prefix[start] + end - start - 1
                if width > max_width:
                    break
                remaining_lines = line_count - line_number - 1
                if len(words) - end < remaining_lines:
                    break
                orphan_cost = max_width ** 2 if end - start == 1 and width < target * 0.6 else 0
                punctuation_bonus = 4 if end < len(words) and words[end - 1].endswith((",", ";", ":", ".", "!", "?")) else 0
                candidate = (cost + (width - target) ** 2 + orphan_cost - punctuation_bonus, (*breaks, end))
                if end not in current or candidate[0] < current[end][0]:
                    current[end] = candidate
        previous = current
    _, breaks = previous[len(words)]
    result = []
    start = 0
    for end in breaks:
        result.append(" ".join(words[start:end]))
        start = end
    return result


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


def token_character_spans(text: str, tokens: list[str] | None = None) -> list[tuple[int, int]] | None:
    """Locate comparison tokens without normalizing the displayed source."""
    tokens = token_texts(text) if tokens is None else tokens
    normalized_parts: list[str] = []
    original_bounds: dict[int, int] = {0: 0}
    normalized_length = 0
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text) and (
            unicodedata.category(text[end]).startswith("M") or text[end] in "\uff9e\uff9f"
        ):
            end += 1
        part = unicodedata.normalize("NFKC", text[start:end])
        normalized_parts.append(part)
        normalized_length += len(part)
        original_bounds[normalized_length] = end
        start = end

    normalized = "".join(normalized_parts)
    for markup_start, markup_end in markup_spans(normalized):
        # A tag attribute such as color="red" must not be found as the word.
        normalized = normalized[:markup_start] + " " * (markup_end - markup_start) + normalized[markup_end:]
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        comparison_token = unicodedata.normalize("NFKC", token)
        start = normalized.find(comparison_token, cursor)
        end = start + len(comparison_token)
        if start not in original_bounds or end not in original_bounds:
            return None
        bounds.append((original_bounds[start], original_bounds[end]))
        cursor = end
    return bounds
