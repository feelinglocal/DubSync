"""Long-episode alignment robustness on small synthetic episodes.

The fixtures reproduce, at a small scale, the failures measured on the cached
45-minute episode 02 stream: an unspoken opening song, a globally offset or
rate-drifted source SRT, and one episode-unique word spoken a few positions
earlier inside a re-worded phrase. A long episode has a fixed alignment cell
budget of roughly 420 cells per source token, so the tests give the small
fixture the same per-token budget instead of a generous one.
"""

from __future__ import annotations

import random

from dubsync import aligner
from dubsync.aligner import align_cues_to_words
from dubsync.alignment_windows import unique_exact_pairs
from dubsync.models import Cue, Word
from dubsync.tokenize import normalized_words, tokenize_cues

LONG_EPISODE_CELLS_PER_TOKEN = 420
_COMMON = ["a", "o", "que", "de", "eu", "nao", "e", "voce", "um", "para", "com", "isso", "se", "me", "na", "ele"]
_SYLLABLES = ["ba", "ce", "di", "fo", "gu", "ha", "ji", "ko", "lu", "ma", "ne", "pi", "qua", "ro", "su", "te", "vi", "xo", "za"]


def _vocabulary(rng: random.Random, size: int) -> list[str]:
    seen: set[str] = set(_COMMON)
    words: list[str] = []
    while len(words) < size:
        candidate = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(3, 4)))
        if candidate not in seen:
            seen.add(candidate)
            words.append(candidate)
    return words


def _synthetic_episode(
    *,
    seed: int = 11,
    cue_count: int = 170,
    song_cues: int = 0,
) -> tuple[list[Cue], list[Word]]:
    """A dubbed episode whose ASR mostly agrees with the SRT, with light improvisation."""

    rng = random.Random(seed)
    vocabulary = _vocabulary(rng, cue_count * 3)
    fresh = iter(_vocabulary(random.Random(seed + 1000), cue_count * 3))
    cues: list[Cue] = []
    words: list[Word] = []
    cursor_ms = 1_000
    for position in range(cue_count):
        index = position + 1
        if position < song_cues:
            lyric = " ".join(rng.choice(vocabulary) for _ in range(5))
            cues.append(Cue(index=index, start_ms=cursor_ms, end_ms=cursor_ms + 2_400, lines=[f"♪{lyric}♪"]))
            cursor_ms += 2_600
            continue
        tokens = [
            rng.choice(vocabulary) if rng.random() < 0.4 else rng.choice(_COMMON)
            for _ in range(rng.randint(3, 7))
        ]
        duration_ms = 420 * len(tokens)
        cues.append(Cue(index=index, start_ms=cursor_ms, end_ms=cursor_ms + duration_ms, lines=[" ".join(tokens)]))
        spoken: list[str] = []
        for token in tokens:
            roll = rng.random()
            if roll < 0.04:
                continue
            if roll < 0.09:
                spoken.append(next(fresh))
                continue
            spoken.append(token)
            if rng.random() < 0.03:
                spoken.append(rng.choice(_COMMON))
        step = duration_ms / 1000.0 / max(1, len(spoken))
        for offset, text in enumerate(spoken):
            start = cursor_ms / 1000.0 + offset * step
            words.append(Word(text=text, start=round(start, 3), end=round(start + step * 0.8, 3), confidence=None))
        cursor_ms += duration_ms + rng.choice((150, 300, 600, 1_200))
    return cues, words


def _retimed(cues: list[Cue], *, shift_ms: int = 0, rate: float = 1.0) -> list[Cue]:
    return [
        cue.with_timing(max(0, int(cue.start_ms * rate) + shift_ms), max(1, int(cue.end_ms * rate) + shift_ms))
        for cue in cues
    ]


def _matched_pairs(result) -> set[tuple[int, int]]:
    return {(match.srt_token_index, match.asr_word_index) for match in result.token_matches}


def _long_episode_budget(monkeypatch, cues: list[Cue]) -> None:
    monkeypatch.setattr(
        aligner,
        "ALIGNMENT_CELL_BUDGET",
        LONG_EPISODE_CELLS_PER_TOKEN * len(tokenize_cues(cues)),
    )


def _assert_resolved(result) -> None:
    assert result.diagnostics.unresolved is False
    assert result.diagnostics.band_limited is False
    assert not any(flag.kind in {"alignment_unresolved", "alignment_band_limited"} for flag in result.flags)


def test_unspoken_opening_song_does_not_collapse_a_long_episode(monkeypatch):
    cues, words = _synthetic_episode(song_cues=24)
    _long_episode_budget(monkeypatch, cues)

    result = align_cues_to_words(cues, words)

    _assert_resolved(result)
    spoken_tokens = [token for token in tokenize_cues(cues) if token.cue_id > 24]
    assert len(result.token_matches) >= 0.85 * len(spoken_tokens)
    assert result.diagnostics.transform_applied is True
    assert abs(result.diagnostics.transform_offset_seconds) < 1.0


def test_offset_and_rate_drifted_source_keeps_the_same_alignment(monkeypatch):
    cues, words = _synthetic_episode(song_cues=24)
    _long_episode_budget(monkeypatch, cues)
    reference = align_cues_to_words(cues, words)
    _assert_resolved(reference)

    for shift_ms, rate in ((30_000, 1.0), (-30_000, 1.0), (120_000, 1.0), (0, 1.04), (0, 0.959), (30_000, 1.04)):
        result = align_cues_to_words(_retimed(cues, shift_ms=shift_ms, rate=rate), words)

        _assert_resolved(result)
        assert len(result.token_matches) >= len(reference.token_matches) - 2, (shift_ms, rate)
        assert len(_matched_pairs(result) & _matched_pairs(reference)) >= len(reference.token_matches) - 4
        assert result.diagnostics.transform_applied is True
        assert abs(result.diagnostics.transform_rate - 1.0 / rate) < 0.01


def test_one_moved_unique_word_does_not_discard_the_whole_alignment(monkeypatch):
    cues, words = _synthetic_episode()
    _long_episode_budget(monkeypatch, cues)
    reference = align_cues_to_words(cues, words)
    _assert_resolved(reference)
    tokens = tokenize_cues(cues)
    by_token = {match.srt_token_index: match.asr_word_index for match in reference.token_matches}
    pairs = unique_exact_pairs(tokens, normalized_words(words))
    span = 8
    token_index, word_index = next(
        (token_index, word_index)
        for token_index, word_index in pairs[len(pairs) // 2 :]
        if all(by_token.get(token_index - offset) == word_index - offset for offset in range(span + 1))
    )
    # The actor says the episode-unique word first, then re-words the rest of the phrase.
    moved = list(words)
    moved[word_index - span] = words[word_index - span].model_copy(update={"text": words[word_index].text})
    for offset in range(span):
        replaced = words[word_index - span + 1 + offset]
        moved[word_index - span + 1 + offset] = replaced.model_copy(update={"text": f"zzq{offset}x"})

    result = align_cues_to_words(cues, moved)

    _assert_resolved(result)
    assert len(result.token_matches) >= len(reference.token_matches) - span - 2
    # The re-worded phrase is reviewable as a divergence instead of the whole episode.
    assert any(word_index - 1 in divergence.asr_word_indices for divergence in result.divergence_spans)
    assert all(len(divergence.cue_ids) <= 3 for divergence in result.divergence_spans)


def test_full_size_offset_episode_aligns_within_a_linear_cell_budget():
    # About 4,500 source tokens, like a 45-minute episode, under the real budget.
    cues, words = _synthetic_episode(seed=5, cue_count=900, song_cues=20)
    cues = _retimed(cues, shift_ms=30_000, rate=1.04)
    tokens = tokenize_cues(cues)
    words_norm = normalized_words(words)
    anchors = aligner._supported_anchor_pairs(unique_exact_pairs(tokens, words_norm))
    reachability = aligner._reachability_centers(tokens, words_norm, None, None, anchors=anchors)

    cells = aligner._band_cell_count(len(tokens), len(words_norm), aligner.BAND_MARGIN, reachability, diagonal=False)
    result = align_cues_to_words(cues, words)

    assert cells <= 200 * len(tokens)
    _assert_resolved(result)
    assert result.diagnostics.transform_applied is True
    assert len(result.token_matches) >= 0.85 * len([token for token in tokens if token.cue_id > 20])


def test_isolated_coincidental_anchor_does_not_bend_the_band():
    pairs = [(10, 12), (18, 21), (25, 27), (40, 300), (52, 55), (60, 62)]

    assert aligner._supported_anchor_pairs(pairs) == [(10, 12), (18, 21), (25, 27), (52, 55), (60, 62)]


def test_unique_exact_pairs_keep_the_longest_consistent_chain():
    cues = [
        Cue(index=1, start_ms=0, end_ms=1_000, lines=["alpha beta gamma"]),
        Cue(index=2, start_ms=1_000, end_ms=2_000, lines=["delta epsilon zeta"]),
    ]
    tokens = tokenize_cues(cues)
    # "zeta" is also heard at the start by coincidence; it is out of order with everything else.
    words_norm = ["zeta", "alpha", "beta", "gamma", "delta", "epsilon"]

    pairs = unique_exact_pairs(tokens, words_norm)

    assert pairs == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
