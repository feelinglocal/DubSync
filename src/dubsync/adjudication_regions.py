from __future__ import annotations

from collections import Counter
from math import isfinite
import unicodedata

from .models import Cue, CueContext, DivergenceSpan, TokenMatch, Word
from .subtitle_annotations import cue_has_bracketed_screen_text, is_bracketed_screen_text_cue
from .text_metrics import join_word_texts
from .tokenize import SRTToken, alphanumeric_signature, normalize_token, tokenize_cues


JOINT_REGION_PREFIX = "joint-"
PROTECTED_SOURCE_PREFIX = "protected-source-"
SPEECH_REPEAT_PREFIX = "speech-repeat-"
SONG_CAPTION_PREFIX = "song-caption-"
_SONG_MARKS = "♪♫"


def is_joint_region(span: DivergenceSpan) -> bool:
    return span.case_id.startswith(JOINT_REGION_PREFIX)


def is_song_caption_cue(cue: Cue) -> bool:
    """A cue that is a song caption as a whole, not dialogue quoting a song."""
    text = cue.plain_text.strip()
    return (
        len(text) > 2 and text[0] in _SONG_MARKS and text[-1] in _SONG_MARKS
        and not cue_has_bracketed_screen_text(cue)
        and bool(alphanumeric_signature(text))
    )


def protect_song_captions(
    spans: list[DivergenceSpan], cues: list[Cue], words: list[Word],
) -> list[DivergenceSpan]:
    """Take song captions out of every divergence an adjudicator may rewrite.

    A dubbed voice track carries no music, so audio inside a caption's span is
    dialogue spoken over the song (or an ASR artifact), never a new lyric. The
    caption tokens become their own source-only case, which the pipeline keeps
    verbatim. Spoken words remain adjudicable: with the dialogue tokens of a
    mixed span, or as a pure insertion next to the caption.
    """
    caption_ids = {cue.index for cue in cues if is_song_caption_cue(cue)}
    if not caption_ids:
        return spans
    tokens = tokenize_cues(cues)
    derived = (JOINT_REGION_PREFIX, PROTECTED_SOURCE_PREFIX, SPEECH_REPEAT_PREFIX, SONG_CAPTION_PREFIX)
    result: list[DivergenceSpan] = []
    for span in spans:
        indices = span.srt_token_indices
        if (
            span.case_id.startswith(derived) or not indices
            or not caption_ids.intersection(span.cue_ids)
            or any(index < 0 or index >= len(tokens) for index in indices)
        ):
            result.append(span)
            continue
        caption_tokens = [index for index in indices if tokens[index].cue_id in caption_ids]
        dialogue_tokens = [index for index in indices if tokens[index].cue_id not in caption_ids]
        if not caption_tokens or (
            dialogue_tokens and dialogue_tokens != list(range(dialogue_tokens[0], dialogue_tokens[-1] + 1))
        ):
            # Dialogue on both sides of a caption cannot be divided without
            # guessing word ownership; the pipeline keeps the complete span.
            result.append(span)
            continue
        audio = (
            list(span.asr_word_indices) if dialogue_tokens
            else _without_neighbour_duplicates(span.asr_word_indices, words)
        )
        if not dialogue_tokens and not audio and not span.asr_word_indices:
            result.append(span)
            continue
        caption_span = span.model_copy(update={
            "case_id": SONG_CAPTION_PREFIX + span.case_id,
            "cue_ids": list(dict.fromkeys(tokens[index].cue_id for index in caption_tokens)),
            "srt_token_indices": caption_tokens,
            "srt_text": join_word_texts(tokens[index].text for index in caption_tokens),
            "asr_text": "", "asr_word_indices": [], "speaker_ids": [], "insertion_token_offset": None,
        })
        if dialogue_tokens:
            speech_span = span.model_copy(update={
                "cue_ids": list(dict.fromkeys(tokens[index].cue_id for index in dialogue_tokens)),
                "srt_token_indices": dialogue_tokens,
                "srt_text": join_word_texts(tokens[index].text for index in dialogue_tokens),
                "insertion_token_offset": None,
            })
        elif audio:
            spoken = [words[index] for index in audio]
            speech_span = span.model_copy(update={
                "cue_ids": [], "srt_token_indices": [], "srt_text": "",
                "asr_word_indices": audio,
                "asr_text": join_word_texts(word.text for word in spoken),
                "start": min(word.start for word in spoken), "end": max(word.end for word in spoken),
                "speaker_ids": sorted({word.speaker_id for word in spoken if word.speaker_id}),
                "insertion_token_offset": None,
            })
        else:
            result.append(caption_span)
            continue
        dialogue_first = bool(dialogue_tokens) and dialogue_tokens[0] < caption_tokens[0]
        result.extend([speech_span, caption_span] if dialogue_first else [caption_span, speech_span])
    return result


def _without_neighbour_duplicates(indices: list[int], words: list[Word]) -> list[int]:
    """Drop a word that repeats the adjacent retained word at the same time.

    A provider can return one spoken word twice with overlapping timestamps
    (MAI: "Queria" / "Queria." at 1228.60 s). The copy that fell into a
    caption's span is not additional speech.
    """
    own = set(indices)
    kept: list[int] = []
    for index in indices:
        if not 0 <= index < len(words):
            return list(indices)
        word = words[index]
        token = normalize_token(word.text)
        duplicate = False
        for neighbour_index in (index - 1, index + 1):
            if neighbour_index in own or not 0 <= neighbour_index < len(words):
                continue
            neighbour = words[neighbour_index]
            shorter = min(word.end - word.start, neighbour.end - neighbour.start)
            overlap = min(word.end, neighbour.end) - max(word.start, neighbour.start)
            if token and token == normalize_token(neighbour.text) and shorter > 0 and overlap >= 0.5 * shorter:
                duplicate = True
        if not duplicate:
            kept.append(index)
    return kept


def split_protected_source_repetitions(
    spans: list[DivergenceSpan], matches: list[TokenMatch], cues: list[Cue],
    tokens: list[SRTToken], words: list[Word], *, protected_cue_ids: set[int],
) -> list[DivergenceSpan]:
    """Separate a distinct repeated utterance from strictly later song captions.

    This only creates an independent audio question. It neither approves ASR
    speech nor changes any retained token match or existing word ownership.
    """
    result: list[DivergenceSpan] = []
    for span in spans:
        pair = _protected_repeat_pair(span, spans, matches, cues, tokens, words, protected_cue_ids)
        result.extend(pair or [span])
    return result


def validated_protected_source_regions(
    spans: list[DivergenceSpan], matches: list[TokenMatch], cues: list[Cue],
    words: list[Word], *, protected_cue_ids: set[int],
) -> dict[str, set[int]]:
    """Revalidate derived branches against real evidence, including on resume.

    A reserved case prefix alone cannot protect arbitrary source or authorize
    new speech. Malformed or orphaned derived branches must be realigned.
    """
    derived = [span for span in spans if span.case_id.startswith((PROTECTED_SOURCE_PREFIX, SPEECH_REPEAT_PREFIX))]
    if not derived:
        return {}
    by_id = {span.case_id: span for span in spans}
    tokens = tokenize_cues(cues)
    result: dict[str, set[int]] = {}
    checked: set[str] = set()
    ignored_fields = {"context_before", "context_after", "prompt_scene_id", "prompt_scene_position"}
    for source in derived:
        if not source.case_id.startswith(PROTECTED_SOURCE_PREFIX):
            continue
        parent_id = source.case_id.removeprefix(PROTECTED_SOURCE_PREFIX)
        speech = by_id.get(SPEECH_REPEAT_PREFIX + parent_id)
        pair = None
        if speech is not None and parent_id not in by_id and len(by_id) == len(spans):
            original = source.model_copy(update={
                "case_id": parent_id, "asr_text": speech.asr_text,
                "asr_word_indices": list(speech.asr_word_indices),
                "start": speech.start, "end": speech.end,
                "confidence": speech.confidence, "speaker_ids": list(speech.speaker_ids),
            })
            pair = _protected_repeat_pair(original, spans, matches, cues, tokens, words, protected_cue_ids)
        if pair is None or any(
            actual.model_dump(exclude=ignored_fields) != expected.model_dump(exclude=ignored_fields)
            for actual, expected in zip((source, speech), pair)
        ):
            raise ValueError("Invalid protected-source adjudication region; resume from align to validate its evidence.")
        result[source.case_id] = set(source.cue_ids)
        checked.update((source.case_id, speech.case_id))
    if checked != {span.case_id for span in derived}:
        raise ValueError("Orphaned protected-source adjudication region; resume from align to validate its evidence.")
    return result


def _only_explicit_song_text(cue: Cue) -> bool:
    lines = [line.strip() for line in cue.lines if line.strip()]
    return bool(lines) and not cue_has_bracketed_screen_text(cue) and all(
        len(line) > 2 and line[0] in "♪♫" and line[-1] in "♪♫"
        and sum(character in "♪♫" for character in line) == 2
        and bool(alphanumeric_signature(line[1:-1]))
        for line in lines
    )


def _protected_repeat_pair(
    span: DivergenceSpan, spans: list[DivergenceSpan], matches: list[TokenMatch],
    cues: list[Cue], tokens: list[SRTToken], words: list[Word], protected: set[int],
) -> list[DivergenceSpan] | None:
    source, audio = span.srt_token_indices, span.asr_word_indices
    if (
        span.case_id.startswith((JOINT_REGION_PREFIX, PROTECTED_SOURCE_PREFIX, SPEECH_REPEAT_PREFIX))
        or not span.cue_ids or not source or len(audio) < 2
        or source != list(range(source[0], source[-1] + 1))
        or audio != list(range(audio[0], audio[-1] + 1))
        or source[0] <= 0 or source[-1] + 1 >= len(tokens)
        or audio[0] <= 0 or audio[-1] + 1 >= len(words)
        or span.insertion_token_offset is not None
    ):
        return None
    cues_by_id = {cue.index: cue for cue in cues}
    if len(cues_by_id) != len(cues) or len(set(span.cue_ids)) != len(span.cue_ids) or any(
        cue_id not in cues_by_id or not _only_explicit_song_text(cues_by_id[cue_id])
        for cue_id in span.cue_ids
    ):
        return None
    footprint = [index for index, token in enumerate(tokens) if token.cue_id in span.cue_ids]
    if (
        footprint != source or tokens != tokenize_cues(cues)
        or list(dict.fromkeys(tokens[index].cue_id for index in source)) != span.cue_ids
        or span.srt_text != " ".join(tokens[index].text for index in source)
        or span.asr_text != " ".join(words[index].text for index in audio)
    ):
        return None
    left_id, right_id = tokens[source[0] - 1].cue_id, tokens[source[-1] + 1].cue_id
    involved_ids = {left_id, right_id, *span.cue_ids}
    if involved_ids & protected or any(
        cue_id not in cues_by_id or cue_has_bracketed_screen_text(cues_by_id[cue_id])
        or any(marker in cues_by_id[cue_id].text for marker in "♪♫")
        for cue_id in (left_id, right_id)
    ):
        return None
    ordered_ids = list(cues_by_id)
    first, last = ordered_ids.index(span.cue_ids[0]), ordered_ids.index(span.cue_ids[-1])
    right_position = ordered_ids.index(right_id)
    if (
        first == 0 or ordered_ids[first - 1] != left_id or right_position <= last
        or any(cue.index not in span.cue_ids and not is_bracketed_screen_text_cue(cue)
               for cue in cues[first:right_position])
        or any(cues_by_id[cue_id].start_ms < 0 or cues_by_id[cue_id].duration_ms <= 0 for cue_id in involved_ids)
        or any(cues_by_id[a].end_ms > cues_by_id[b].start_ms for a, b in zip(span.cue_ids, span.cue_ids[1:]))
    ):
        return None
    prior = [index for index, token in enumerate(tokens) if token.cue_id == left_id]
    if len(prior) < 2 or prior != list(range(prior[0], source[0])):
        return None
    by_token = {match.srt_token_index: match for match in matches}
    source_matches = Counter(match.srt_token_index for match in matches)
    audio_matches = Counter(match.asr_word_index for match in matches)
    span_sources = Counter(index for item in spans for index in item.srt_token_indices)
    span_audio = Counter(index for item in spans for index in item.asr_word_indices)
    retained = [by_token.get(index) for index in [*prior, source[-1] + 1]]
    if any(match is None for match in retained):
        return None
    if (
        any(source_matches[index] or span_sources[index] != 1 for index in source)
        or any(audio_matches[index] or span_audio[index] != 1 for index in audio)
        or any(span_sources[index] for index in [*prior, source[-1] + 1])
        or [match.asr_word_index for match in retained[:-1]] != list(range(audio[0] - len(prior), audio[0]))
        or retained[-1].asr_word_index != audio[-1] + 1
    ):
        return None
    for match in retained:
        index = match.asr_word_index
        if (
            index < 0 or index >= len(words) or match.score != 1.0
            or source_matches[match.srt_token_index] != 1 or audio_matches[index] != 1
            or span_audio[index]
            or match.cue_id != tokens[match.srt_token_index].cue_id
            or tokens[match.srt_token_index].normalized != normalize_token(words[index].text)
        ):
            return None
    before, after = words[retained[-2].asr_word_index], words[retained[-1].asr_word_index]
    spoken = [words[index] for index in audio]
    prior_words = [words[match.asr_word_index] for match in retained[:-1]]
    signature = [tokens[index].normalized for index in prior]
    evidence = [*prior_words, *spoken, after]
    if (
        len(set(signature)) < 2
        or [normalize_token(item.text) for item in spoken] != signature
        or span.left_anchor_cue_id != left_id or span.right_anchor_cue_id != right_id
        or span.left_anchor_end != before.end or span.right_anchor_start != after.start
        or span.left_anchor_speaker_id != before.speaker_id or span.right_anchor_speaker_id != after.speaker_id
        or span.start != spoken[0].start or span.end != spoken[-1].end
        or any(len(alphanumeric_signature(item.text)) != 1 or any(marker in item.text for marker in "♪♫")
               or not isfinite(item.start) or not isfinite(item.end) or item.start < 0
               or not 0 < item.end - item.start <= 2.0
               or item.confidence is not None and (not isfinite(item.confidence) or not 0.8 <= item.confidence <= 1)
               for item in evidence)
        or any(not 0 <= following.start - previous.end <= 0.2 + 1e-9
               for previous, following in zip(evidence[:-2], evidence[1:-1]))
        or spoken[-1].end >= min(cues_by_id[cue_id].start_ms / 1000 for cue_id in span.cue_ids)
        or after.start < max(cues_by_id[cue_id].end_ms / 1000 for cue_id in span.cue_ids)
        or after.start <= spoken[-1].end
    ):
        return None
    speakers = {item.speaker_id for item in evidence if item.speaker_id}
    speakers.update(speaker for speaker in [*span.speaker_ids, *[cues_by_id[cue_id].speaker_id for cue_id in involved_ids]] if speaker)
    if len(speakers) > 1:
        return None
    source_span = span.model_copy(update={
        "case_id": PROTECTED_SOURCE_PREFIX + span.case_id,
        "asr_text": "", "asr_word_indices": [], "speaker_ids": [], "confidence": 0.0,
        "start": min(cues_by_id[cue_id].start_ms for cue_id in span.cue_ids) / 1000,
        "end": max(cues_by_id[cue_id].end_ms for cue_id in span.cue_ids) / 1000,
    })
    speech_span = span.model_copy(update={
        "case_id": SPEECH_REPEAT_PREFIX + span.case_id,
        "cue_ids": [], "srt_text": "", "srt_token_indices": [],
        "context_before": [CueContext(cue_id=cue.index, text=cue.plain_text,
            start=cue.start_ms / 1000, end=cue.end_ms / 1000) for cue in cues[max(0, first - 2):first]],
        "context_after": [CueContext(cue_id=cue.index, text=cue.plain_text,
            start=cue.start_ms / 1000, end=cue.end_ms / 1000) for cue in cues[first:first + 2]],
    })
    return [source_span, speech_span]


def join_isolated_anchor_regions(
    spans: list[DivergenceSpan],
    matches: list[TokenMatch],
    cues: list[Cue],
    tokens: list[SRTToken],
    words: list[Word],
    *,
    protected_cue_ids: set[int],
) -> list[DivergenceSpan]:
    """Ask for fresh adjudication of one ambiguous match between two edits.

    Global matches and their ownership remain untouched. A joint case includes
    the formerly retained token explicitly, so its old neighboring decisions
    cannot approve either its spelling or a new acoustic owner.
    """
    by_token = {match.srt_token_index: match for match in matches}
    match_counts = Counter(match.cue_id for match in matches)
    token_counts = Counter(token.cue_id for token in tokens)
    cues_by_id = {cue.index: cue for cue in cues}
    result: list[DivergenceSpan] = []
    position = 0
    while position < len(spans):
        joined = None
        if position + 1 < len(spans):
            joined = _join_pair(
                spans[position], spans[position + 1], by_token, match_counts,
                token_counts, cues_by_id, tokens, words, protected_cue_ids,
            )
        result.append(joined or spans[position])
        position += 2 if joined is not None else 1
    return result


def _join_pair(
    left: DivergenceSpan, right: DivergenceSpan, by_token: dict[int, TokenMatch],
    match_counts: Counter[int], token_counts: Counter[int], cues_by_id: dict[int, Cue],
    tokens: list[SRTToken], words: list[Word], protected: set[int],
) -> DivergenceSpan | None:
    if is_joint_region(left) or is_joint_region(right):
        return None
    for span in (left, right):
        source, audio = span.srt_token_indices, span.asr_word_indices
        if (
            not span.cue_ids or not source or not audio
            or source != list(range(source[0], source[-1] + 1))
            or audio != list(range(audio[0], audio[-1] + 1))
            or source[0] < 0 or source[-1] >= len(tokens)
            or audio[0] < 0 or audio[-1] >= len(words)
            or alphanumeric_signature(span.srt_text) != [tokens[index].normalized for index in source]
            or alphanumeric_signature(span.asr_text) != alphanumeric_signature(" ".join(words[index].text for index in audio))
        ):
            return None
    bridge_source = left.srt_token_indices[-1] + 1
    bridge_audio = left.asr_word_indices[-1] + 1
    if right.srt_token_indices[0] != bridge_source + 1 or right.asr_word_indices[0] != bridge_audio + 1:
        return None
    bridge = by_token.get(bridge_source)
    if bridge is None or bridge.asr_word_index != bridge_audio or bridge.score != 1.0:
        return None
    token, word = tokens[bridge_source], words[bridge_audio]
    # A literal retained word is not reopened merely because it is short.
    # This narrow case requires an accent-folding collision such as é / E.
    literal_word = "".join(character for character in unicodedata.normalize("NFC", word.text).casefold() if character.isalnum())
    if (
        len(token.normalized) != 1 or token.normalized != normalize_token(word.text)
        or unicodedata.normalize("NFC", token.text).casefold() == literal_word
        or match_counts[token.cue_id] != 1
        or left.cue_ids[-1] != token.cue_id or right.cue_ids[0] != token.cue_id
        or len(left.asr_word_indices) < 2
    ):
        return None
    source_indices = list(range(left.srt_token_indices[0], right.srt_token_indices[-1] + 1))
    audio_indices = list(range(left.asr_word_indices[0], right.asr_word_indices[-1] + 1))
    cue_ids = list(dict.fromkeys(tokens[index].cue_id for index in source_indices))
    # A retained tail, the isolated cue, and at most one fully consumed neighbor.
    if not 2 <= len(cue_ids) <= 3 or cue_ids[1] != token.cue_id:
        return None
    before = by_token.get(source_indices[0] - 1)
    after = by_token.get(source_indices[-1] + 1)
    if (
        before is None or after is None or before.score != 1.0 or after.score != 1.0
        or before.cue_id != cue_ids[0] or after.cue_id in cue_ids
        or before.asr_word_index != audio_indices[0] - 1
        or after.asr_word_index != audio_indices[-1] + 1
        or match_counts[cue_ids[0]] < 2
        or left.left_anchor_cue_id != before.cue_id or right.right_anchor_cue_id != after.cue_id
    ):
        return None
    ordered_ids = list(cues_by_id)
    first_position = ordered_ids.index(cue_ids[0])
    if ordered_ids[first_position:first_position + len(cue_ids) + 1] != [*cue_ids, after.cue_id]:
        return None
    covered_counts = Counter(tokens[index].cue_id for index in source_indices)
    if covered_counts[cue_ids[0]] >= token_counts[cue_ids[0]] or any(
        covered_counts[cue_id] != token_counts[cue_id] for cue_id in cue_ids[1:]
    ):
        return None
    target_tokens = [index for index, item in enumerate(tokens) if item.cue_id == after.cue_id]
    target_matches = [by_token.get(index) for index in target_tokens]
    if (
        len(target_matches) < 2
        or any(match is None or match.score != 1.0 for match in target_matches)
        or [match.asr_word_index for match in target_matches] != list(range(after.asr_word_index, after.asr_word_index + len(target_matches)))
        or before.asr_word_index < 0 or target_matches[-1].asr_word_index >= len(words)
    ):
        return None
    involved = [*cue_ids, after.cue_id]
    if set(involved) & protected or any(
        cue_has_bracketed_screen_text(cues_by_id[cue_id])
        or any(marker in cues_by_id[cue_id].text for marker in ("♪", "♫"))
        for cue_id in involved
    ):
        return None
    evidence = [words[before.asr_word_index], *[words[index] for index in audio_indices],
                *[words[match.asr_word_index] for match in target_matches]]
    if any(
        not alphanumeric_signature(item.text)
        or any(marker in item.text for marker in ("♪", "♫"))
        or not isfinite(item.start) or not isfinite(item.end)
        or item.start < 0 or not 0 < item.end - item.start <= 2.0
        or item.confidence is not None and item.confidence < 0.8
        for item in evidence
    ):
        return None
    known_speakers = {item.speaker_id for item in evidence if item.speaker_id}
    known_speakers.update(speaker for speaker in (
        *left.speaker_ids, *right.speaker_ids, left.left_anchor_speaker_id,
        left.right_anchor_speaker_id, right.left_anchor_speaker_id, right.right_anchor_speaker_id,
        *[cues_by_id[cue_id].speaker_id for cue_id in involved],
    ) if speaker)
    if len(known_speakers) > 1:
        return None
    # The sole phrase gap must be immediately before the ambiguous bridge;
    # every other word, including both retained anchors, must join closely.
    bridge_position = 1 + len(left.asr_word_indices)
    for index in range(1, len(evidence)):
        gap = evidence[index].start - evidence[index - 1].end
        if not (gap >= 0.8 if index == bridge_position else 0 <= gap <= 0.2):
            return None
    spoken = [words[index] for index in audio_indices]
    confidences = [item.confidence for item in spoken if item.confidence is not None]
    return DivergenceSpan(
        case_id=f"{JOINT_REGION_PREFIX}{left.case_id}--{right.case_id}",
        cue_ids=cue_ids, srt_token_indices=source_indices, asr_word_indices=audio_indices,
        srt_text=" ".join(tokens[index].text for index in source_indices),
        asr_text=" ".join(item.text for item in spoken),
        start=spoken[0].start, end=spoken[-1].end,
        confidence=min(confidences) if confidences else 0.0,
        speaker_ids=sorted({item.speaker_id for item in spoken if item.speaker_id}),
        left_anchor_cue_id=before.cue_id, right_anchor_cue_id=after.cue_id,
        left_anchor_end=evidence[0].end, right_anchor_start=words[after.asr_word_index].start,
        left_anchor_speaker_id=evidence[0].speaker_id,
        right_anchor_speaker_id=words[after.asr_word_index].speaker_id,
        context_before=list(left.context_before), context_after=list(right.context_after),
    )
