from __future__ import annotations

import re
from collections.abc import Iterable

# MAI-Transcribe diarizes each transcription chunk on its own and prefixes its
# labels with the chunk ("chunk_3:1"). Scribe labels ("speaker_4") are valid
# for the whole episode.
_CHUNK_SCOPED_SPEAKER = re.compile(r"^(chunk_\d+):")


def speakers_known_different(left: str | None, right: str | None) -> bool:
    """Whether two speaker labels provably name different actors.

    A missing label proves nothing. Labels from different chunk scopes are
    unrelated: the same actor gets another label in the next chunk, so their
    relation is unknown and never evidence of a speaker change.
    """
    if not left or not right or left == right:
        return False
    return _label_scope(left) == _label_scope(right)


def has_known_different_speakers(speaker_ids: Iterable[str | None]) -> bool:
    labels = [label for label in dict.fromkeys(speaker_ids) if label]
    return any(
        speakers_known_different(left, right)
        for position, left in enumerate(labels)
        for right in labels[position + 1:]
    )


def _label_scope(label: str) -> str | None:
    match = _CHUNK_SCOPED_SPEAKER.match(label)
    return match.group(1) if match else None
