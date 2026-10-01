import pytest

from dubsync.models import Cue
from dubsync.source_language import infer_source_language


def cues(text):
    return [Cue(index=1, start_ms=0, end_ms=1000, lines=[text])]


@pytest.mark.parametrize(("text", "expected"), [
    ("Você não está aqui porque eu estou falando com vocês. Então também pode fazer alguma coisa para ela.", "pt"),
    ("Ich weiß nicht warum du heute hier bist. Wir haben auch etwas für euch und können noch darüber sprechen.", "de"),
    ("I know that you are here because we have something for them. They would never tell us what happened.", "en"),
    ("私はあなたと一緒にここで待っています。でも何も分かりません。", "ja"),
    ("你好我们正在这里等候您的回复谢谢", None),
    ("Não.", None),
    ("Okay. Luke. Donny.", None),
    ("Você não está aqui porque eu estou falando. Ich weiß nicht warum du heute hier bist. Wir haben auch etwas für euch.", None),
])
def test_source_language_requires_sustained_unambiguous_evidence(text, expected):
    assert infer_source_language(cues(text)) == expected


def test_screen_text_and_song_lyrics_do_not_choose_dialogue_language():
    assert infer_source_language(cues("[Você não está aqui porque eu estou falando com vocês então também alguma coisa]")
                                 + [Cue(index=2, start_ms=2000, end_ms=4000, lines=["♪ Ich weiß nicht warum du heute hier bist wir haben auch etwas für euch ♪"])]) is None


def test_repetition_of_one_distinctive_word_is_insufficient():
    assert infer_source_language(cues("Você " * 50)) is None
