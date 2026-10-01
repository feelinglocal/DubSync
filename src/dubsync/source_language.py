"""Conservative source-language hints for otherwise automatic transcription.

This is deliberately an abstaining hint, not a language gate. Short, mixed or
unrecognised scripts keep the provider's automatic detection. Explicit user
language always takes precedence at the caller.
"""
from collections import Counter
import re
from collections.abc import Sequence

from .adjudication_regions import is_song_caption_cue
from .models import Cue
from .subtitle_annotations import speech_text_for_alignment


_MARKERS = {
    "pt": frozenset("você vocês não então também estou está estão nós isso uma para pelo pela fazer porque alguma alguma coisa muito aqui com pode tenho você esta foi".split()),
    "de": frozenset("ich nicht weiß warum heute hier bist wir haben auch etwas für euch können noch darüber sprechen das ist sie sind mein meine einen aber doch jetzt".split()),
    "en": frozenset("the this that with you are because we have something them they would never tell what happened your don't didn't will just know here".split()),
    "es": frozenset("usted ustedes no entonces también estoy está están nosotros eso una para hacer porque alguna cosa mucho aquí con puede tengo pero esto fue".split()),
    "fr": frozenset("vous nous pas avec pourquoi suis êtes sont dans pour mais aussi quelque chose peux cette c'est".split()),
    "it": frozenset("sono sei siete non perché anche questo questa quello quella cosa possiamo voglio grazie molto".split()),
}
_ALL_COUNTS = Counter(word for words in _MARKERS.values() for word in words)
_EXCLUSIVE = {language: words - {word for word, count in _ALL_COUNTS.items() if count > 1}
              for language, words in _MARKERS.items()}


def infer_source_language(cues: Sequence[Cue]) -> str | None:
    text = " ".join(speech_text_for_alignment(cue) for cue in cues if not is_song_caption_cue(cue))
    letters = [char for char in text if char.isalpha()]
    kana = sum("\u3040" <= char <= "\u30ff" for char in letters)
    if kana >= 12 and kana / max(1, len(letters)) >= 0.3:
        return "ja"
    tokens = Counter(re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text.casefold()))
    ranked = sorted(((sum(min(3, tokens[word]) for word in words),
                      sum(word in tokens for word in words), language)
                     for language, words in _EXCLUSIVE.items()), reverse=True)
    score, distinct, language = ranked[0]
    runner_up = ranked[1][0]
    if ranked[1][1] >= 3 and runner_up >= score * 0.1:
        return None
    if distinct >= 5 and score >= 8 and score >= 3 * max(1, runner_up):
        return language
    return None
