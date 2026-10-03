"""Small, deterministic final-display segmentation using owned word timing."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil, floor, isfinite
import re
import unicodedata

from .cue_segmentation import _ends_sentence
from .models import Cue, QCFlag, Word
from .recue import timing_evidence_issue
from .style_profile import StyleProfile
from .subtitle_annotations import bracketed_screen_text_spans, cue_has_bracketed_screen_text, is_bracketed_screen_text_cue
from .text_metrics import (
    _JAPANESE_PUNCTUATION, _NONSTARTING_KANA, _can_break_between, _is_unspaced_script, _text_clusters,
    contains_character_level_script, display_width, is_character_level_script, join_word_texts,
    token_character_spans, token_texts, wrap_visual_width,
)
from .tokenize import alphanumeric_signature


@dataclass(frozen=True)
class OutputSegmentation:
    cues: list[Cue]
    cue_word_indices: dict[int, list[int]]
    flags: list[QCFlag]
    expansions: dict[int, list[int]]
    caption_pages: dict[int, list[dict[str, object]]] = field(default_factory=dict)


_GLUE_WORDS = frozenset("a an the of to in on at by for with from and but or if as is are was were be been being can could would should will must this that these those de do da dos das no na nos nas um uma o os as por ao aos pela pelo e mas que se não só un une le la les des du au aux et pour à en el los las del al y und der die das dem den ein eine einer eines im am zum zur von mit an auf zu".split())
_CONJUNCTIONS = frozenset("and but or because while although e mas porém porque quando embora y pero porque et mais und aber weil obwohl".split())
_PHRASES = ("por causa de", "de acordo com", "a fim de", "em vez de", "não só", "mas também", "as well as", "in order to", "because of", "such as")
_CLOSERS = "\"'’”»)]}」』）］｝】"
_OPENERS = "\"'‘“«([{「『（［｛【"
_MARKUP = re.compile(r"</?[^>\n]+>|{\\[^}\n]+}")
_GERMAN_POSSESSIVES = frozenset(
    root + ending for root in ("mein", "dein", "sein", "ihr", "unser", "euer", "eur")
    for ending in ("", "e", "en", "em", "er", "es")
)
_GERMAN_DETERMINERS = _GERMAN_POSSESSIVES | frozenset(
    "der die das dem den des ein eine einen einem einer eines diese diesen diesem dieser dieses".split()
)
_GERMAN_ADJECTIVE = re.compile(r"[a-zäöüß]+(?:e|en|em|er|es)$")
# A line-initial dash opens one speaker's turn in a dialogue cue.
_DIALOGUE_DASHES = "-–—"
_TURN_MARKER = re.compile(r"[-–—]\s")
# A no-break, narrow no-break or figure space binds its neighbours (French
# "ici ?"); every other whitespace run may become a line break.
_BREAKABLE_SPACE = re.compile(r"[^\S\u00a0\u202f\u2007]+")
_DISPLAY_UNIT = re.compile(r"(?:[^\s]|[\u00a0\u202f\u2007])+")
_OPENING_UNIT = frozenset(_OPENERS)
_TRAILING_UNIT = frozenset(_CLOSERS + ".,!?;:…%")
# Without a tokenizer these kana pairs are one word, one ending or one compound particle.
_PARTICLE_COMPOUNDS = frozenset(
    "です でし ます ませ まし すね すよ すか でも では ても ては には にも とは とも のは ので のに へは へも".split())
_JAPANESE_CLAUSE_ENDS = ("けれど", "けど", "から", "ので", "のに", "ながら", "たら", "れば", "ても", "でも")


def _bare(unit: str) -> str:
    unit = unit[2:] if _TURN_MARKER.match(unit) else unit
    return unit.strip(_OPENERS + _CLOSERS + ".,!?;:…。、\u00a0\u202f\u2007 ").casefold()


def _unit_spans(text: str) -> list[tuple[int, int]]:
    """Breakable display units of ``text``, as positions that keep its own separators.

    A dash that opens a turn (at a line start or after a sentence end) and an
    opening mark stay with the word they introduce. A unit of closing
    punctuation, and a dash inside a sentence, stay with the word before
    them: a line never starts with a dash that would read as a new speaker.
    """
    spans: list[tuple[int, int]] = []
    for match in _DISPLAY_UNIT.finditer(text):
        unit = match.group()
        if not unit.strip():
            continue
        if spans:
            previous = text[spans[-1][0]:spans[-1][1]]
            inner_dash = unit in _DIALOGUE_DASHES and not (
                "\n" in text[spans[-1][1]:match.start()] or _ends_sentence(previous))
            if previous in _DIALOGUE_DASHES or set(previous) <= _OPENING_UNIT or set(unit) <= _TRAILING_UNIT \
                    or inner_dash:
                spans[-1] = (spans[-1][0], match.end())
                continue
        spans.append(match.span())
    return spans


def _joins_unspaced(left: str, right: str) -> bool:
    return any(_is_unspaced_script(char) or char in _JAPANESE_PUNCTUATION for char in (left, right))


def _joined_lines(lines: list[str]) -> str:
    """Authored lines as one text; Japanese lines join without an inserted space."""
    text = ""
    for line in (line.strip() for line in lines):
        if line:
            text += line if not text or _joins_unspaced(text[-1], line[0]) else " " + line
    return text


def _display_text(text: str) -> str:
    """One display text: a line break or space run becomes one space, but an
    ideographic or other non-ASCII space keeps its own character."""
    def separator(match: re.Match[str]) -> str:
        return "".join(char for char in match.group() if not char.isascii()) or " "
    return _BREAKABLE_SPACE.sub(separator, _joined_lines(text.splitlines()))


def _dialogue_turns(lines: list[str]) -> list[str] | None:
    """Complete dialogue turns, when a later line opens another speaker's turn with a dash."""
    turns: list[str] = []
    for line in (line.strip() for line in lines):
        if not line:
            continue
        if turns and not _TURN_MARKER.match(line):
            turns[-1] = _joined_lines([turns[-1], line])
        else:
            turns.append(line)
    return turns if len(turns) > 1 else None


def _cue_text(cue: Cue) -> tuple[str, frozenset[int]]:
    """The cue's words as one text, and the offsets where its authored lines start.

    Japanese lines join without an inserted space. Each dialogue turn keeps
    a line of its own: turns are separated by a newline in the text.
    """
    lines = [line.strip() for line in cue.text.splitlines() if line.strip()]
    turns = _dialogue_turns(lines) is not None
    text, starts = "", set()
    for line in lines:
        if text:
            if turns and _TURN_MARKER.match(line):
                text += "\n"
            elif not _joins_unspaced(text[-1], line[0]):
                text += " "
            starts.add(len(text))
        text += line
    return text, frozenset(starts)


def _display_lines(text: str, width: int) -> list[str]:
    """Each complete dialogue turn on a line of its own; otherwise semantic wrapping."""
    turns = [turn.strip() for turn in text.split("\n") if turn.strip()]
    return turns if len(turns) > 1 else wrap_semantic_lines(text, width)


def _turn_lines(turns: list[str], limit: int) -> list[str]:
    """One line per turn; with more turns than lines, whole turns share a line."""
    if len(turns) <= limit:
        return list(turns)
    if limit == 1:
        return [" ".join(turns)]
    position = min(range(1, len(turns)), key=lambda p: (
        max(display_width(" ".join(turns[:p])), display_width(" ".join(turns[p:]))), p))
    return [" ".join(turns[:position]), " ".join(turns[position:])]


def _german_noun_attachment(units: list[str], position: int, normalized: list[str]) -> bool:
    """Recognize a determiner/adjective group only with its capitalized head."""
    def adjective(index):
        word = units[index].lstrip(_OPENERS + _DIALOGUE_DASHES + " ")
        return word[:1].islower() and _GERMAN_ADJECTIVE.fullmatch(normalized[index]) is not None
    right = position
    while right < len(units) and adjective(right):
        right += 1
    if right == len(units) or not units[right].lstrip(_OPENERS + _DIALOGUE_DASHES + " ")[:1].isupper():
        return False
    left = position - 1
    while left >= 0 and normalized[left] not in _GERMAN_DETERMINERS and adjective(left):
        left -= 1
    return left >= 0 and normalized[left] in _GERMAN_DETERMINERS


def _legal_break(units: list[str], position: int) -> bool:
    if position == len(units):
        return True
    previous = units[position - 1].rstrip(_CLOSERS)
    if previous.endswith((",", ";", ":", "、")) or _ends_sentence(previous, next_text=units[position]):
        return True
    if previous.endswith("."):
        return False  # An abbreviation, ordinal or initial stays with its name.
    if _bare(previous) in _GLUE_WORDS:
        return False
    normalized = [_bare(unit) for unit in units]
    if _german_noun_attachment(units, position, normalized):
        return False
    for phrase in _PHRASES:
        parts = phrase.split()
        for start in range(max(0, position - len(parts) + 1), position):
            if start + len(parts) > position and normalized[start:start + len(parts)] == parts:
                return False
    return True


def _break_cost(left: str, right: str) -> int:
    previous = left.rstrip().rstrip(_CLOSERS)
    if _ends_sentence(previous, next_text=right):
        return 0
    if previous.endswith((";", ":")):
        return 4
    if previous.endswith((",", "、")):
        return 10
    if right.split() and _bare(right.split()[0]) in _CONJUNCTIONS:
        return 20
    return 70


def wrap_semantic_lines(text: str, max_width: int) -> list[str]:
    """Prefer complete phrases and punctuation while retaining every token.

    Lines keep the text's own separators: a no-break or ideographic space is
    never rewritten as an ASCII space, and authored Japanese lines join
    without one.
    """
    text = _display_text(text)
    spans = _unit_spans(text)
    if not spans:
        return []
    if len(spans) == 1:
        # Never add discretionary hyphens to a long Latin word.
        return _unspaced_lines(text, max_width) if contains_character_level_script(text) else [text]
    if not _MARKUP.search(text) and contains_character_level_script(text):
        # An over-wide Japanese phrase may break at its own legal clusters.
        refined = []
        for start, end in spans:
            unit = text[start:end]
            if display_width(unit) <= max_width or not contains_character_level_script(unit) or " " in unit:
                refined.append((start, end))
                continue
            for piece in _unspaced_lines(unit, max_width):
                refined.append((start, start + len(piece)))
                start += len(piece)
        spans = refined
    units = [text[start:end] for start, end in spans]
    widths = [display_width(unit) for unit in units]
    gaps = [display_width(text[left[1]:right[0]]) for left, right in zip(spans, spans[1:])] + [0]
    greedy_count, current = 1, 0
    for position, width in enumerate(widths):
        gap = gaps[position - 1] if current else 0
        if current and current + gap + width > max_width:
            greedy_count += 1
            current = width
        else:
            current += gap + width
    if greedy_count == 1 or len(units) > 128 or any(width > max_width for width in widths) or _MARKUP.search(text):
        return _greedy_lines(text, spans, max_width)
    prefix = [0]
    for width, gap in zip(widths, gaps):
        prefix.append(prefix[-1] + width + gap)
    target = (prefix[-1] - greedy_count + 1) / greedy_count
    for count in [greedy_count]:
        states: dict[int, tuple[float, list[int]]] = {0: (0, [])}
        for line in range(count):
            following: dict[int, tuple[float, list[int]]] = {}
            for start, (cost, cuts) in states.items():
                for end in range(start + 1, len(units) + 1):
                    width = prefix[end] - prefix[start] - gaps[end - 1]
                    if width > max_width:
                        break
                    if len(units) - end < count - line - 1:
                        continue
                    boundary = (_break_cost(text[spans[start][0]:spans[end - 1][1]], text[spans[end][0]:])
                                if end < len(units) else 0)
                    attachment_cost = 500 if not _legal_break(units, end) else 0
                    candidate = (cost + (width - target) ** 2 + boundary * 3 + attachment_cost, [*cuts, end])
                    if end not in following or candidate[0] < following[end][0]:
                        following[end] = candidate
            states = following
        if len(units) in states:
            result, start = [], 0
            for end in states[len(units)][1]:
                result.append(text[spans[start][0]:spans[end - 1][1]])
                start = end
            return result
    return _greedy_lines(text, spans, max_width)


def wrap_generated_lines(text: str, max_width: int) -> list[str]:
    """Wrap a cue joined from ASR words in audio-to-SRT generation.

    Unspaced Japanese/Chinese text takes the same balanced, kinsoku-legal
    layout as synchronized output; spaced text keeps the balanced word wrap.
    """
    if len(text.split()) == 1 and contains_character_level_script(text):
        return wrap_semantic_lines(text, max_width)
    return wrap_visual_width(text, max_width)


def _display_clusters(text: str) -> list[str]:
    """Retain kana marks and common emoji graphemes as indivisible units."""
    clusters: list[str] = []
    for cluster in _text_clusters(text):
        first = ord(cluster[0])
        regional_pair = (0x1F1E6 <= first <= 0x1F1FF and clusters
                         and len(clusters[-1]) == 1 and 0x1F1E6 <= ord(clusters[-1]) <= 0x1F1FF)
        attached = (cluster[0] == "\u200d" or 0x1F3FB <= first <= 0x1F3FF
                    or 0xE0020 <= first <= 0xE007F or unicodedata.category(cluster[0]) == "Mc")
        if clusters and (attached or clusters[-1].endswith("\u200d") or regional_pair):
            clusters[-1] += cluster
        else:
            clusters.append(cluster)
    return clusters


def _unspaced_break_cost(clusters: list[str], end: int) -> int:
    if end == len(clusters):
        return 0
    previous = unicodedata.normalize("NFKC", clusters[end - 1])[-1]
    if previous in "。！？、；：.!?,;:":
        return -96
    # These are preferences, not a tokenizer: balance and legal kinsoku
    # boundaries still apply. を is a particularly clear phrase boundary;
    # a connective て/で keeps the following verb phrase together. です or
    # には is one word, not a particle followed by another phrase.
    if previous + clusters[end][0] in _PARTICLE_COMPOUNDS:
        return 64
    if previous == "を":
        return -64
    if previous in "はがにへともやのてで":
        return -32
    if previous in _NONSTARTING_KANA:
        return 64
    script = _japanese_run_script(previous)
    if script is not None and script == _japanese_run_script(unicodedata.normalize("NFKC", clusters[end])[0]):
        # A kanji or katakana run is mostly one word or compound.
        return 32
    return 0


def _japanese_run_script(character: str) -> str | None:
    name = unicodedata.name(character, "")
    if character in "々〆" or name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
        return "kanji"
    return "katakana" if "KATAKANA" in name and character != "・" else None


def _unspaced_lines(text: str, max_width: int) -> list[str]:
    """Balance short character-level lines without widening the style limit."""
    if display_width(text) <= max_width:
        return [text]
    if _MARKUP.search(text):
        return wrap_visual_width(text, max_width)
    clusters = _display_clusters(text)
    prefix = [0]
    for cluster in clusters:
        prefix.append(prefix[-1] + display_width(cluster))
    legal = [True, *(_can_break_between(left, right) for left, right in zip(clusters, clusters[1:])), True]
    greedy, start = [], 0
    while start < len(clusters):
        end = start + 1
        while end < len(clusters) and prefix[end + 1] - prefix[start] <= max_width:
            end += 1
        fitted = end
        while end > start and not legal[end]:
            end -= 1
        if end == start:
            # Impossible widths retain the whole punctuation/grapheme group;
            # ordinary line-length QC reports the resulting visible overflow.
            end = fitted
            while not legal[end]:
                end += 1
        greedy.append("".join(clusters[start:end]))
        start = end
    if max_width <= 0 or len(greedy) <= 1 or len(greedy) > 8 or len(clusters) > 256:
        return greedy
    for count in range(ceil(prefix[-1] / max_width), len(greedy) + 1):
        target = prefix[-1] / count
        states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0, ())}
        for line in range(count):
            following: dict[int, tuple[float, tuple[int, ...]]] = {}
            for start, (cost, cuts) in states.items():
                for end in range(start + 1, len(clusters) + 1):
                    width = prefix[end] - prefix[start]
                    if width > max_width:
                        break
                    if not legal[end] or len(clusters) - end < count - line - 1:
                        continue
                    orphan = max_width ** 2 if end - start == 1 and width < target * .6 else 0
                    candidate = (cost + (width - target) ** 2 + orphan + _unspaced_break_cost(clusters, end),
                                 (*cuts, end))
                    if end not in following or candidate[0] < following[end][0]:
                        following[end] = candidate
            states = following
        if len(clusters) in states:
            lines, start = [], 0
            for end in states[len(clusters)][1]:
                lines.append("".join(clusters[start:end]))
                start = end
            return lines
    return greedy


def _greedy_lines(text: str, spans: list[tuple[int, int]], width: int) -> list[str]:
    lines: list[tuple[int, int]] = []
    for start, end in spans:
        if lines and display_width(text[lines[-1][0]:end]) > width:
            lines.append((start, end))
        elif lines:
            lines[-1] = (lines[-1][0], end)
        else:
            lines.append((start, end))
    return [text[start:end] for start, end in lines]


def compact_lines(text: str, limit: int, width: int) -> list[str]:
    """Keep the existing envelope when safe child speech timing is unavailable.

    Dialogue turns keep a line each while they fit the line limit; otherwise
    whole turns share a line. A turn is never divided.
    """
    turns = _dialogue_turns(text.splitlines())
    if turns is not None:
        return _turn_lines(turns, limit)
    stripped = _display_text(text)
    lines = wrap_semantic_lines(stripped, width)
    if len(lines) <= limit:
        return lines
    if limit == 1:
        return [stripped]
    spans = _unit_spans(stripped)
    units = [stripped[start:end] for start, end in spans]

    def halves(position: int) -> tuple[str, str]:
        return stripped[:spans[position - 1][1]], stripped[spans[position][0]:]

    candidates = [position for position in range(1, len(units)) if _legal_break(units, position)]
    if candidates:
        position = min(candidates, key=lambda p: (
            abs(display_width(halves(p)[0]) - display_width(halves(p)[1])) + _break_cost(*halves(p)) * 2,
            p,
        ))
        return list(halves(position))
    midpoint = max(1, len(lines) // 2)
    return [join_word_texts(lines[:midpoint]), join_word_texts(lines[midpoint:])]


def paginate_annotation_lines(lines: list[str], width: int, limit: int) -> list[list[str]]:
    """Turn a known visual caption into complete, ordered display pages."""
    text = "\n".join(lines)
    spans = bracketed_screen_text_spans(text)
    pages: list[list[str]] = []
    current: list[str] = []
    for start, end in spans:
        body = text[start + 1:end - 1].strip()
        chunks = _visual_chunks(body, max(1, width - 2), limit)
        for chunk in chunks:
            rendered = wrap_semantic_lines(chunk, max(1, width - 2)) or [""]
            rendered[0] = "[" + rendered[0]
            rendered[-1] += "]"
            if current and len(current) + len(rendered) > limit:
                pages.append(current)
                current = []
            current.extend(rendered)
    if current:
        pages.append(current)
    return pages or [compact_lines(text, limit, width)]


def _visual_chunks(text: str, width: int, limit: int) -> list[str]:
    """Page boundaries favour whole clauses over balanced character counts."""
    wrapped = wrap_semantic_lines(text, width)
    if len(wrapped) <= limit:
        return [text]
    text = _display_text(text)
    spans = _unit_spans(text)
    units = [text[start:end] for start, end in spans]
    if len(units) <= 1 or len(units) > 128:
        return [join_word_texts(wrapped[position:position + limit])
                for position in range(0, len(wrapped), limit)]
    states: dict[int, tuple[int, float, list[str]]] = {0: (0, 0, [])}
    for end in range(1, len(units) + 1):
        if not _legal_break(units, end):
            continue
        for start, (count, cost, chunks) in list(states.items()):
            if start >= end:
                continue
            chunk = text[spans[start][0]:spans[end - 1][1]]
            if len(wrap_semantic_lines(chunk, width)) > limit:
                continue
            penalty = _break_cost(chunk, text[spans[end][0]:]) if end < len(units) else 0
            candidate = (count + 1, cost + penalty, [*chunks, chunk])
            if end not in states or candidate[:2] < states[end][:2]:
                states[end] = candidate
    if len(units) in states:
        return states[len(units)][2]
    return [join_word_texts(wrapped[position:position + limit])
            for position in range(0, len(wrapped), limit)]


def _minimum_display_prefix(text: str, width: int) -> list[float]:
    """A monotone lower bound for the width that line wrapping must consume.

    Whitespace collapses as in the wrapper. An oversized token counts as at
    most one full line, preserving its existing overflow/kinsoku behaviour.
    This may admit extra candidates; it never rejects a fitting old result.
    """
    prefix = [0.0]
    for match in re.finditer(r"\s+|\S+", text):
        unit = match.group()
        if unit.isspace():
            prefix.append(prefix[-1] + 1)
            prefix.extend([prefix[-1]] * (len(unit) - 1))
        else:
            scale = min(1.0, width / max(1, display_width(unit)))
            for char in unit:
                prefix.append(prefix[-1] + display_width(char) * scale)
    return prefix


def _japanese_script(char: str) -> str | None:
    code = ord(char)
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x323AF:
        return "kanji"
    if 0x30A1 <= code <= 0x30FA or 0x30FC <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9F:
        return "katakana"
    return None


def _legal_unspaced_cut(text: str, char: int, authored: bool) -> bool:
    """Japanese break legality for a cue boundary inside unspaced text.

    Kinsoku always applies. Without a tokenizer a kanji compound or a
    katakana word is one word: only the customer's own line break may
    divide one, and a cut never follows a small っ or precedes ん.
    """
    left, right = text[char - 1], text[char]
    if unicodedata.combining(right) or right in "\uff9e\uff9f" or not _can_break_between(left, right):
        return False
    if authored:
        return True
    if left in "っッｯ" or right in "んンﾝ":
        return False
    if left.isalnum() and right.isalnum() and not is_character_level_script(left) \
            and not is_character_level_script(right):
        return False  # Inside a Latin word or number.
    return _japanese_script(left) is None or _japanese_script(left) != _japanese_script(right)


def _unspaced_cut_cost(text: str, char: int, authored: bool) -> int:
    """Prefer punctuation, the customer's own breaks, clause ends and particles."""
    cost = _break_cost(text[:char], text[char:])
    if authored:
        return min(cost, 10)
    previous, following = text[char - 1], text[char]
    if text[:char].endswith(_JAPANESE_CLAUSE_ENDS):
        phrase = 20
    elif previous == "を":
        phrase = 30
    elif previous in "はがにへともやのてで" and previous + following not in _PARTICLE_COMPOUNDS:
        phrase = 35 if _japanese_script(following) else 45
    elif _japanese_script(following) and not _japanese_script(previous):
        phrase = 55  # A content word begins after kana.
    elif _japanese_script(previous) and not _japanese_script(following):
        phrase = 80  # A kanji stem usually continues in kana.
    else:
        phrase = 70
    return min(cost, phrase + (30 if previous in _NONSTARTING_KANA else 0))


def _timed_parts(cue: Cue, words: list[Word], indices: list[int], profile: StyleProfile, limit: int,
                 min_parts: int) -> tuple[list[list[str]], list[list[int]], list[tuple[int, int]]] | None:
    text, authored = _cue_text(cue)
    if cue_has_bracketed_screen_text(cue) or _MARKUP.search(text) or not indices:
        return None
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(words) for index in indices):
        return None
    ordered = sorted(indices, key=lambda index: (words[index].start, words[index].end, index))
    selected = [words[index] for index in ordered]
    if any(not isfinite(word.start) or not isfinite(word.end) or word.end <= word.start for word in selected):
        return None
    if timing_evidence_issue(cue, selected) is not None:
        return None
    tokens = token_texts(text)
    spans = token_character_spans(text, tokens)
    if spans is None or alphanumeric_signature(text) != [token for word in selected for token in alphanumeric_signature(word.text)]:
        return None
    unspaced = contains_character_level_script(text)
    unit_spans = _unit_spans(text)
    units = [text[start:end] for start, end in unit_spans]
    unit_starts = [start for start, _ in unit_spans]
    unit_at = {start: position for position, start in enumerate(unit_starts)}
    # A cut can only occur between complete provider words, and between
    # complete displayed tokens. Punctuation-only words stay with a neighbour;
    # a dialogue dash moves with the turn it opens.
    positions, token_count = [(0, 0, cue.start_ms, 0)], 0
    for position, word in enumerate(selected[:-1], start=1):
        token_count += len(alphanumeric_signature(word.text))
        if not token_count or token_count >= len(spans):
            continue
        left_end = max(item.end for item in selected[:position]) * 1000
        right_start = min(item.start for item in selected[position:]) * 1000
        boundary = profile.snap_floor(right_start)
        if boundary < left_end:
            boundary = ceil(left_end - 1e-8)
        if boundary > floor(right_start + 1e-8) or not cue.start_ms < boundary < cue.end_ms:
            continue
        char = spans[token_count][0]
        while char and text[char - 1] in _OPENERS:
            char -= 1
        unit_start = unit_starts[bisect_right(unit_starts, char) - 1] if unit_starts and unit_starts[0] <= char else char
        if unit_start < char and text[unit_start] in _DIALOGUE_DASHES and not text[unit_start + 1:char].strip():
            char = unit_start
        left, right = text[:char].rstrip(), text[char:].lstrip()
        if not left or not right:
            continue
        if text[char - 1].isspace():
            if char not in unit_at or not _legal_break(units, unit_at[char]):
                continue
            # A new dialogue turn is the best boundary; in Japanese a space
            # is the customer's own phrase break.
            cost = 0 if text[char - 1] == "\n" else _break_cost(left, right)
            cost = min(cost, 10) if unspaced else cost
        elif unspaced and _legal_unspaced_cut(text, char, char in authored):
            cost = _unspaced_cut_cost(text, char, char in authored)
        else:
            continue  # Unspaced text may cut at provider boundaries; a Latin token may not.
        positions.append((position, char, boundary, cost))
    positions.append((len(ordered), len(text), cue.end_ms, 0))
    minimum_width = _minimum_display_prefix(text, profile.max_chars_per_line)
    starts, ends = [], []
    for _, char, _, _ in positions:
        first, last = char, char
        while first < len(text) and text[first].isspace():
            first += 1
        while last > 0 and text[last - 1].isspace():
            last -= 1
        starts.append(minimum_width[first])
        ends.append(minimum_width[last])
    capacity = limit * profile.max_chars_per_line + limit - 1
    fitting: dict[tuple[int, int], bool] = {}

    def turn_edge(char: int) -> bool:
        return char in (0, len(text)) or text[char - 1] == "\n"

    def display(first_char: int, last_char: int) -> list[str]:
        chunk = text[first_char:last_char].strip()
        if not chunk:
            return []
        breaks = [first_char, *sorted(char for char in authored if first_char < char < last_char), last_char]
        own = [text[left:right].strip() for left, right in zip(breaks, breaks[1:])]
        # A child made of the customer's own whole lines keeps them while they fit.
        if ((first_char == 0 or first_char in authored) and (last_char == len(text) or last_char in authored)
                and len(own) <= limit and all(display_width(line) <= profile.max_chars_per_line for line in own)):
            return own
        return _display_lines(chunk, profile.max_chars_per_line)

    def fits(start: int, end: int) -> bool:
        if (start, end) not in fitting:
            first_char, last_char = positions[start][1], positions[end][1]
            lines = display(first_char, last_char)
            # Dialogue turns share a display only whole, each on a line within
            # the width; a turn is divided only where it has a display to itself.
            fitting[start, end] = bool(lines) and len(lines) <= limit and (
                "\n" not in text[first_char:last_char].strip() or turn_edge(first_char) and turn_edge(last_char)
                and all(display_width(line) <= profile.max_chars_per_line for line in lines))
        return fitting[start, end]

    def solve(target: float | None) -> list[tuple[int, float, list[int]]]:
        # Keep the same count/cost/tie ordering, but visit only states at the
        # candidate start instead of rescanning every earlier position's states.
        states: list[dict[int, tuple[float, list[int]]]] = [{} for _ in positions]
        states[0][0] = (0, [])
        first_possible = 0
        for end in range(1, len(positions)):
            while first_possible < end and ends[end] - starts[first_possible] > capacity + 1e-7:
                first_possible += 1
            for start in range(first_possible, end):
                if not states[start]:
                    continue
                first_word, first_char, start_ms, _ = positions[start]
                last_word, last_char, end_ms, cut_cost = positions[end]
                if not fits(start, end):
                    continue
                if any(word.start * 1000 < start_ms - 1e-7 or word.end * 1000 > end_ms + 1e-7
                       for word in selected[first_word:last_word]):
                    continue
                boundary_cost = cut_cost if end < len(positions) - 1 else 0
                spoken_end_ms = max(word.end for word in selected[first_word:last_word]) * 1000
                duration_cost = max(0, profile.min_cue_dur * 1000 - (spoken_end_ms - start_ms)) / 10
                balance_cost = 0 if target is None else 100 * (
                    (display_width(text[first_char:last_char].strip()) - target) / profile.max_chars_per_line) ** 2
                for count, (cost, cuts) in states[start].items():
                    candidate = (cost + boundary_cost + duration_cost + balance_cost, [*cuts, end])
                    if count + 1 not in states[end] or candidate[0] < states[end][count + 1][0]:
                        states[end][count + 1] = candidate
        return [(count, cost, cuts) for count, (cost, cuts) in states[-1].items() if count >= min_parts]

    options = solve(None)
    if not options:
        return None
    if unspaced:
        # Japanese phrase costs are only preferences: among the fewest
        # children, balanced displays win over the earliest feasible cut.
        fewest = min(count for count, _, _ in options)
        options = [option for option in solve(display_width(text) / fewest) if option[0] == fewest]
    _, _, cuts = min(options, key=lambda item: (item[0], item[1], item[2]))
    chunks, groups, intervals, start = [], [], [], 0
    for end in cuts:
        chunks.append(display(positions[start][1], positions[end][1]))
        groups.append(ordered[positions[start][0]:positions[end][0]])
        first_ms = positions[start][2]
        spoken_end = max(words[index].end for index in groups[-1]) * 1000
        padded_end = max(profile.snap_ceil(spoken_end + profile.tail_ms),
                         profile.snap_ceil(first_ms + profile.min_cue_dur * 1000))
        # Readability padding is explicit. A large silence before the next
        # phrase is not an acoustic reason to prolong this child over it.
        last_ms = min(positions[end][2], padded_end)
        intervals.append((first_ms, max(first_ms + 1, last_ms)))
        start = end
    return chunks, groups, intervals


def split_crowded_output_cues(cues: list[Cue], words: list[Word],
                             cue_word_indices: Mapping[int, list[int]], profile: StyleProfile,
                             *, max_lines: int | None = None, min_parts: int = 1,
                             protected_cue_ids: set[int] | None = None,
                             enforce_width: bool = True,
                             reserved_cue_ids: set[int] | None = None,
                             hold_unfitting: bool = False) -> OutputSegmentation:
    """Apply the final two-line ceiling without estimating speech timestamps.

    With ``enforce_width`` false the width is the customer's own (derived from
    the source file): a cue within the line limit keeps its lines whatever
    their width, which ordinary line-length QC reports. Only a cue over the
    line limit is reflowed or split.

    Dialogue turns (lines opened by a dash) keep a line each; a timed split
    falls between turns or inside a turn that has the display to itself.

    ``reserved_cue_ids`` are identities a new child must never take, such as
    source cues removed earlier in the run or cues a finding still names.
    With ``hold_unfitting`` a cue that can be neither divided nor fitted to
    the lines within the width, without joining dialogue turns, keeps its own
    lines; the caller makes room for them.
    """
    limit = min(2, profile.max_lines_per_cue, max_lines if max_lines is not None else 2)
    if limit < 1 or min_parts < 1:
        raise ValueError("display line and part limits must be positive")
    ownership = {index: list(indices) for index, indices in cue_word_indices.items()}
    counts = Counter(index for indices in ownership.values() for index in indices)
    protected = protected_cue_ids or set()
    next_id = max([0, *ownership, *(cue.index for cue in cues), *(reserved_cue_ids or ())]) + 1
    output, flags, expansions, visual_pages = [], [], {}, {}
    for cue in cues:
        fits_width = not enforce_width or all(display_width(line) <= profile.max_chars_per_line for line in cue.lines)
        if len(cue.text.splitlines()) <= limit and fits_width and min_parts == 1:
            output.append(cue)
            continue
        text, _ = _cue_text(cue)
        compact = _display_lines(text, profile.max_chars_per_line)
        turns_fit = "\n" not in text or all(display_width(line) <= profile.max_chars_per_line for line in compact)
        if len(compact) <= limit and turns_fit and min_parts == 1 and not cue_has_bracketed_screen_text(cue):
            reflowed = cue.with_lines(compact)
            output.append(reflowed)
            flags.append(QCFlag(kind="output_line_limit_reflow", cue_ids=[cue.index], severity="info",
                                message="The complete spoken phrase fits the available display lines; its authored breaks were reflowed without changing its timing or wording.",
                                old_text=cue.text, new_text=reflowed.text,
                                start=cue.start_ms / 1000, end=cue.end_ms / 1000))
            continue
        visual = is_bracketed_screen_text_cue(cue) and not ownership.get(cue.index) and not any(mark in cue.text for mark in "♪♫")
        replacements: list[Cue] = []
        if visual:
            pages = paginate_annotation_lines(cue.lines, profile.max_chars_per_line, limit)
            duration = cue.duration_ms
            if duration >= len(pages) and len(pages) > 1:
                weights = [max(1, display_width(" ".join(page))) for page in pages]
                elapsed, boundaries = 0, [cue.start_ms]
                for position, weight in enumerate(weights[:-1], start=1):
                    elapsed += weight
                    proposed = profile.snap_floor(cue.start_ms + duration * elapsed / sum(weights))
                    boundaries.append(max(boundaries[-1] + 1, min(proposed, cue.end_ms - len(pages) + position)))
                boundaries.append(cue.end_ms)
                records = []
                for position, page in enumerate(pages):
                    index = cue.index if not position else next_id
                    next_id += bool(position)
                    replacements.append(cue.model_copy(update={"index": index, "start_ms": boundaries[position],
                                                              "end_ms": boundaries[position + 1], "lines": page}))
                    ownership[index] = []
                    records.append({"page": position + 1, "lines": list(page), "display_cue_ids": [index],
                                    "display_intervals": [[boundaries[position], boundaries[position + 1]]]})
                visual_pages[cue.index] = records
        else:
            indices = ownership.get(cue.index, [])
            requested = max(min_parts, 2)
            parts = _timed_parts(cue, words, indices, profile, limit, requested) if cue.index not in protected and not any(counts[index] > 1 for index in indices) else None
            if parts is not None:
                chunks, groups, intervals = parts
                for position, (chunk, group) in enumerate(zip(chunks, groups, strict=True)):
                    index = cue.index if not position else next_id
                    next_id += bool(position)
                    replacements.append(cue.model_copy(update={"index": index, "start_ms": intervals[position][0],
                                                              "end_ms": intervals[position][1],
                                                              "lines": chunk}))
                    ownership[index] = list(group)
        if replacements:
            output.extend(replacements)
            expansions[cue.index] = [item.index for item in replacements]
            flags.append(QCFlag(kind="output_line_limit_split", cue_ids=expansions[cue.index], severity="info",
                                message="Display was divided at complete visual phrases or exact owned-word boundaries, retaining every word.",
                                old_text=cue.text, new_text="\n\n".join(item.text for item in replacements),
                                start=cue.start_ms / 1000, end=cue.end_ms / 1000))
        else:
            reflowed = cue.with_lines(compact_lines(cue.text, limit, profile.max_chars_per_line))
            turns = _dialogue_turns(cue.text.splitlines())
            if hold_unfitting and (len(reflowed.lines) < len(turns or ()) or any(
                    display_width(line) > profile.max_chars_per_line for line in reflowed.lines)):
                output.append(cue)  # Never one over-wide line or joined turns: the caller makes room.
                continue
            if not enforce_width and reflowed.lines == cue.lines:
                output.append(cue)  # Nothing was reflowed: the customer's lines stay without a layout flag.
                continue
            output.append(reflowed)
            flags.append(QCFlag(kind="output_line_limit_reflow", cue_ids=[cue.index], severity="info",
                                message="A safe spoken split was unavailable. Text was reflowed within its existing interval; width overflow remains visible to style QC.",
                                old_text=cue.text, new_text=reflowed.text, start=cue.start_ms / 1000, end=cue.end_ms / 1000))
    output.sort(key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    return OutputSegmentation(output, ownership, flags, expansions, visual_pages)


def expand_output_flags(flags: list[QCFlag], expansions: Mapping[int, list[int]]) -> list[QCFlag]:
    """Keep structural review references attached to every actual child cue."""
    expanded: list[QCFlag] = []
    actual = {index: children for index, children in expansions.items() if len(children) > 1}
    for flag in flags:
        cue_ids = list(dict.fromkeys(child for index in flag.cue_ids for child in actual.get(index, [index])))
        expanded.append(flag.model_copy(update={"cue_ids": cue_ids}) if cue_ids != flag.cue_ids else flag)
    return expanded
