"""A source-derived style keeps the customer's one- and two-line layouts.

The final display pass enforces the two-line ceiling. A line that is only
wider than the width inferred from the customer's own file is a style
finding, not a reason to re-break or split the cue; an explicit style still
enforces its width.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import wave

import pytest
import yaml

from dubsync import pipeline
from dubsync.annotation_composition import compose_bracketed_annotations
from dubsync.models import Cue, Word
from dubsync.semantic_output import split_crowded_output_cues
from dubsync.srt_io import parse_srt_text, write_srt
from dubsync.style_profile import StyleProfile, derive_style_profile
from dubsync.subtitle_annotations import is_bracketed_screen_text_cue, speech_text_for_alignment
from dubsync.text_metrics import display_width
from dubsync.tokenize import alphanumeric_signature
from dubsync.transcription import generate_srt_from_audio

_LINE_LIMIT_KINDS = {"output_line_limit_reflow", "output_line_limit_split"}

_GERMAN_FILLER = [
    ["Wir gehen jetzt los.", "Komm schon mit."],
    ["Das ist doch klar."],
    ["Ich weiß es nicht.", "Frag lieber ihn."],
    ["Heute Abend essen wir", "bei meiner Mutter."],
    ["Wo warst du gestern?"],
]
# Valid customer layouts that the width pass re-broke or split (matrix
# testing-001 cue 73, testing-008 cue 43, testing-002 cue 40, German #3,
# testing3-026 cue 40).
_GERMAN_TARGETS = [
    ["kannst du dir deinen", "Geburtstagswunsch überlegen?"],
    ["- Halt die Klappe, du Tr*ttel!", "- Ah!"],
    ["Äh, sie haben mich seit", "drei Monaten nicht bezahlt."],
    ["schob mein Stiefsohn Rafael"],
    ["die letzte", "Wirtschaftskrise überstand,"],
]
_JAPANESE_FILLER = [["そうだな"], ["分かった"], ["行くぞ", "早くしろ"], ["ありがとう"], ["何だって"]]
# Two of the six delivered Japanese cues: single 28-column customer lines.
_JAPANESE_TARGETS = [["大口を叩いてくれるじゃねぇか"], ["テーブルをひっくり返したのは"]]


def _customer_cues(filler: list[list[str]], targets: list[list[str]], filler_count: int = 60) -> list[Cue]:
    layouts = [filler[position % len(filler)] for position in range(filler_count)] + targets
    return [Cue(index=position + 1, start_ms=1000 + position * 2000, end_ms=2600 + position * 2000, lines=lines)
            for position, lines in enumerate(layouts)]


def _spoken_units(cue: Cue) -> list[str]:
    if " " not in cue.plain_text:
        return list("".join(cue.lines))  # Character-level script: one provider word per character.
    return [unit for unit in cue.plain_text.split() if any(char.isalnum() for char in unit)]


def _exact_words(cues: list[Cue]) -> tuple[list[Word], dict[int, list[int]]]:
    words: list[Word] = []
    ownership: dict[int, list[int]] = {}
    for cue in cues:
        units = _spoken_units(cue)
        step = (cue.duration_ms / 1000 - .2) / len(units)
        ownership[cue.index] = list(range(len(words), len(words) + len(units)))
        words.extend(Word(text=unit, start=round(cue.start_ms / 1000 + .05 + position * step, 3),
                          end=round(cue.start_ms / 1000 + .05 + (position + .8) * step, 3))
                     for position, unit in enumerate(units))
    return words, ownership


@pytest.mark.parametrize("filler,targets", [(_GERMAN_FILLER, _GERMAN_TARGETS),
                                            (_JAPANESE_FILLER, _JAPANESE_TARGETS)], ids=["german", "japanese"])
def test_source_derived_width_never_rebreaks_or_splits_a_valid_customer_cue(filler, targets):
    cues = _customer_cues(filler, targets)
    profile = derive_style_profile(cues)
    words, ownership = _exact_words(cues)
    overwide = [cue for cue in cues if any(display_width(line) > profile.max_chars_per_line for line in cue.lines)]
    assert profile.max_chars_per_line == 26 and profile.max_lines_per_cue == 2
    assert [cue.lines for cue in overwide] == [lines for lines in targets
                                               if max(map(display_width, lines)) > 26]
    assert len(overwide) >= 2

    result = split_crowded_output_cues(cues, words, ownership, profile, enforce_width=False)

    assert len(result.cues) == len(cues)
    assert all(delivered is source for delivered, source in zip(result.cues, cues, strict=True))
    assert not result.flags and not result.expansions and not result.caption_pages
    assert result.cue_word_indices == ownership
    # The same call with the width enforced is the explicit-style behaviour.
    enforced = split_crowded_output_cues(cues, words, ownership, profile)
    assert {flag.kind for flag in enforced.flags} <= _LINE_LIMIT_KINDS
    assert {cue_id for flag in enforced.flags for cue_id in flag.cue_ids} >= {cue.index for cue in overwide}


_VOCABULARY = ("ja nein vielleicht Geburtstagswunsch Wirtschaftskrise überlegen, bezahlt. Stiefsohn "
               "obrigado, você empresa, escândalo. the quick meeting, tomorrow. unbelievable").split()
_KANA = "そうだな分かったありがとう大口を叩いてテーブルをひっくり返した"


def _ink(text: str) -> str:
    return "".join(char for char in text if not char.isspace() and char not in "[]")


def _random_line(rng: random.Random, japanese: bool) -> str:
    if japanese:
        return "".join(rng.choice(_KANA) for _ in range(rng.randint(2, 22)))
    return " ".join(rng.choice(_VOCABULARY) for _ in range(rng.randint(1, 9)))


@pytest.mark.parametrize("seed", range(8))
def test_any_one_or_two_line_cue_keeps_its_lines_and_any_longer_cue_keeps_its_words(seed):
    rng = random.Random(seed)
    cues = []
    for position in range(120):
        japanese = rng.random() < .25
        line_count = rng.choice([1, 1, 2, 2, 2, 3, 4])
        lines = [_random_line(rng, japanese) for _ in range(line_count)]
        if rng.random() < .1:
            lines = [f"[{line}]" for line in lines]  # Screen text follows the same line rule.
        cues.append(Cue(index=position + 1, start_ms=1000 + position * 4000, end_ms=4600 + position * 4000,
                        lines=lines))
    spoken = [cue for cue in cues if not is_bracketed_screen_text_cue(cue)]
    words, ownership = _exact_words(spoken)
    if seed % 2:
        words, ownership = [], {}  # Held or source-timed cues have no owned words.
    profile = StyleProfile(max_chars_per_line=rng.choice([12, 20, 26, 40]), min_cue_dur=.1)

    result = split_crowded_output_cues(cues, words, ownership, profile, enforce_width=False)

    delivered = {cue.index: cue for cue in result.cues}
    assert all(len(cue.text.splitlines()) <= 2 for cue in result.cues)
    flagged = {cue_id for flag in result.flags for cue_id in flag.cue_ids}
    for cue in cues:
        children = result.expansions.get(cue.index, [cue.index])
        if len(cue.lines) <= 2:
            assert delivered[cue.index] is cue and children == [cue.index] and cue.index not in flagged
            continue
        # Only a cue over the line ceiling is changed, and it loses nothing.
        assert _ink("".join(delivered[child].text for child in children)) == _ink(cue.text)
        assert delivered[children[0]].start_ms == cue.start_ms and delivered[children[-1]].end_ms <= cue.end_ms
        assert cue.index in flagged
        if cue.index in ownership:
            assert Counter(index for child in children for index in result.cue_word_indices[child]) == Counter(
                ownership[cue.index])
    assert {flag.kind for flag in result.flags} <= _LINE_LIMIT_KINDS
    # Screen-text composition without any caption over speech is the same pass.
    composed = compose_bracketed_annotations(cues, ownership, words=words, profile=profile, enforce_width=False)
    assert composed.cues == result.cues and composed.cue_word_indices == result.cue_word_indices


def test_a_call_without_a_customer_layout_still_enforces_the_width():
    # Generation and an explicit style have no source-derived width to respect.
    source = Cue(index=1, start_ms=1000, end_ms=3400, lines=["schob mein Stiefsohn Rafael"])
    words, ownership = _exact_words([source])
    result = split_crowded_output_cues([source], words, ownership, StyleProfile(max_chars_per_line=26))
    assert [cue.lines for cue in result.cues] == [["schob mein Stiefsohn", "Rafael"]]
    assert [flag.kind for flag in result.flags] == ["output_line_limit_reflow"]


def test_overwide_caption_line_shares_a_display_with_one_spoken_line_without_splitting_the_speech():
    # Speech plus caption is two lines: no width alone divides the spoken cue.
    speech = Cue(index=1, start_ms=1000, end_ms=4000, lines=["Vamos embora daqui agora mesmo, por favor."])
    caption = Cue(index=2, start_ms=1500, end_ms=3500,
                  lines=["[Hospital Central da Cidade de São Paulo]"])
    words, ownership = _exact_words([speech])
    profile = StyleProfile(max_chars_per_line=30, min_cue_dur=.1)

    composed = compose_bracketed_annotations([speech, caption], ownership, words=words, profile=profile,
                                             enforce_width=False)

    assert [cue.lines for cue in composed.cues] == [[*speech.lines, *caption.lines]]
    assert (composed.cues[0].start_ms, composed.cues[0].end_ms) == (1000, 4000)
    assert not composed.expansions and not composed.flags
    assert composed.cue_word_indices == ownership
    # An explicit width divides the same speech at owned-word boundaries.
    explicit = compose_bracketed_annotations([speech, caption], ownership, words=words, profile=profile)
    assert len(explicit.cues) > 1 and all(len(cue.lines) <= 2 for cue in explicit.cues)


def test_speech_that_kept_its_lines_carries_no_line_limit_flag():
    # Two captions want two spoken children; this held cue has no owned words to divide.
    speech = Cue(index=1, start_ms=1000, end_ms=4000, lines=["Vamos embora daqui."])
    captions = [Cue(index=2, start_ms=1000, end_ms=4000, lines=["[Hospital Central]"]),
                Cue(index=3, start_ms=1000, end_ms=4000, lines=["[Quarto 12]"])]
    profile = StyleProfile(max_chars_per_line=47, min_cue_dur=.5)

    composed = compose_bracketed_annotations([speech, *captions], {1: []}, words=[], profile=profile,
                                             enforce_width=False)

    assert [cue.lines for cue in composed.cues] == [["Vamos embora daqui.", "[Hospital Central] [Quarto 12]"]]
    assert [flag.kind for flag in composed.flags] == ["annotation_line_limit_reflow"]
    # With the width enforced the same pass still records that it kept the lines.
    enforced = compose_bracketed_annotations([speech, *captions], {1: []}, words=[], profile=profile)
    assert [cue.lines for cue in enforced.cues] == [cue.lines for cue in composed.cues]
    assert sorted(flag.kind for flag in enforced.flags) == ["annotation_line_limit_reflow", "output_line_limit_reflow"]


def _sync(tmp_path, cues: list[Cue], **options):
    source, audio, fixture, config = (tmp_path / name for name in (
        "episode.srt", "episode.wav", "words.json", "providers.yaml"))
    source.write_text(write_srt(cues), encoding="utf-8")
    audio.write_bytes(b"RIFF....WAVEfmt ")
    words, _ = _exact_words(cues)
    fixture.write_text(json.dumps({"words": [word.model_dump() for word in words]}, ensure_ascii=False),
                       encoding="utf-8")
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")
    return pipeline.sync_episode(source, audio, tmp_path / "output.srt", tmp_path / "work",
                                 providers_path=config, no_llm=True, **options)


@pytest.mark.parametrize("mode", ["fresh", "cache", "rebuild", "verify"])
@pytest.mark.parametrize("filler,targets", [(_GERMAN_FILLER, _GERMAN_TARGETS),
                                            (_JAPANESE_FILLER, _JAPANESE_TARGETS)], ids=["german", "japanese"])
def test_default_sync_delivers_every_customer_layout_byte_for_byte(tmp_path, mode, filler, targets):
    cues = _customer_cues(filler, targets)

    result = _sync(tmp_path, cues)
    first = result.output_srt.read_bytes()
    if mode != "fresh":
        result = _sync(tmp_path, cues, **({} if mode == "cache" else {"resume": mode}))
        assert result.output_srt.read_bytes() == first

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.lines for cue in delivered] == [cue.lines for cue in cues]
    report = result.report
    assert not [flag for flag in report["flags"] if flag["kind"] in _LINE_LIMIT_KINDS]
    assert not [change for change in report["changes"] if change["kind"] in _LINE_LIMIT_KINDS]
    # The customer's wide lines stay a style finding about their own text.
    wide = {cue.index for cue in cues if any(display_width(line) > 26 for line in cue.lines)}
    assert {issue["cue_id"] for issue in report["style_issues"] if issue["kind"] == "line_length"} == wide
    assert not [item for item in report["review"] if "line_length" in json.dumps(item)]
    assert any(item["kind"] == "style:line_length" for item in report["diagnostics"])


@pytest.mark.parametrize("mode", ["fresh", "verify", "verify_from_saved_style"])
@pytest.mark.parametrize("filler,targets", [(_GERMAN_FILLER, _GERMAN_TARGETS),
                                            (_JAPANESE_FILLER, _JAPANESE_TARGETS)], ids=["german", "japanese"])
def test_web_maximum_lines_option_keeps_every_one_or_two_line_customer_cue(tmp_path, mode, filler, targets):
    # web/jobs.py "Maximum lines per cue: 2": the source-derived profile with only the line count set.
    # The customer chose a line count, not a width, so the inferred width changes nothing.
    cues = _customer_cues(filler, targets)
    profile = derive_style_profile(cues).model_copy(update={"max_lines_per_cue": 2})

    result = _sync(tmp_path, cues, style_profile=profile)
    first = result.output_srt.read_bytes()
    if mode != "fresh":
        result = _sync(tmp_path, cues, resume="verify", **({"style_profile": profile} if mode == "verify" else {}))
        assert result.output_srt.read_bytes() == first

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert [cue.lines for cue in delivered] == [cue.lines for cue in cues]
    assert not [flag for flag in result.report["flags"]
                if flag["kind"] in _LINE_LIMIT_KINDS or flag["kind"].startswith("sync_cue_line_limit")]
    wide = {cue.index for cue in cues if any(display_width(line) > profile.max_chars_per_line for line in cue.lines)}
    assert wide and {issue["cue_id"] for issue in result.report["style_issues"] if issue["kind"] == "line_length"} == wide


def test_web_maximum_lines_option_still_reduces_a_three_line_customer_cue(tmp_path):
    crowded = ["Wir haben die Arbeit beendet.", "Jetzt gehen wir nach Hause.", "Bring bitte die Schlüssel mit."]
    cues = _customer_cues(_GERMAN_FILLER, [_GERMAN_TARGETS[0], crowded], filler_count=12)
    cues[-1] = cues[-1].with_timing(cues[-1].start_ms, cues[-1].start_ms + 6000)
    profile = derive_style_profile(cues).model_copy(update={"max_lines_per_cue": 2})

    result = _sync(tmp_path, cues, style_profile=profile)

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert all(len(cue.lines) <= 2 for cue in delivered)
    assert [cue.lines for cue in delivered[:len(cues) - 1]] == [cue.lines for cue in cues[:-1]]
    assert " ".join(cue.plain_text for cue in delivered[len(cues) - 1:]) == " ".join(crowded)


@pytest.mark.parametrize("mode", ["fresh", "verify", "verify_from_saved_style"])
def test_explicit_narrower_style_still_reflows_a_wide_customer_line(tmp_path, mode):
    cues = _customer_cues(_GERMAN_FILLER, [["schob mein Stiefsohn Rafael"]])
    profile = derive_style_profile(cues).model_copy(update={"max_chars_per_line": 24})

    result = _sync(tmp_path, cues, style_profile=profile)
    first = result.output_srt.read_bytes()
    if mode != "fresh":
        # A resume without the option reads the saved style, which is stricter than the source.
        result = _sync(tmp_path, cues, resume="verify", **({"style_profile": profile} if mode == "verify" else {}))
        assert result.output_srt.read_bytes() == first

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert delivered[-1].lines == ["schob mein Stiefsohn", "Rafael"]
    assert [flag["cue_ids"] for flag in result.report["flags"]
            if flag["kind"] == "output_line_limit_reflow"] == [[cues[-1].index]]
    assert [cue.lines for cue in delivered[:-1]] == [cue.lines for cue in cues[:-1]]


@pytest.mark.parametrize("mode", ["fresh", "verify"])
def test_default_sync_still_reduces_a_three_line_customer_cue_without_losing_words(tmp_path, mode):
    crowded = ["Wir haben die Arbeit beendet.", "Jetzt gehen wir nach Hause.", "Bring bitte die Schlüssel mit."]
    cues = _customer_cues(_GERMAN_FILLER, [_GERMAN_TARGETS[0], crowded], filler_count=12)
    cues[-1] = cues[-1].with_timing(cues[-1].start_ms, cues[-1].start_ms + 6000)

    result = _sync(tmp_path, cues)
    if mode == "verify":
        result = _sync(tmp_path, cues, resume="verify")

    delivered = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert all(len(cue.lines) <= 2 for cue in delivered)
    assert [cue.lines for cue in delivered[:len(cues) - 1]] == [cue.lines for cue in cues[:-1]]
    reduced = delivered[len(cues) - 1:]
    assert " ".join(cue.plain_text for cue in reduced) == " ".join(crowded)
    kinds = [flag["kind"] for flag in result.report["flags"] if flag["kind"] in _LINE_LIMIT_KINDS]
    assert len(kinds) == 1


def test_generation_still_wraps_every_cue_to_its_style_width(tmp_path):
    # Generated cues have no customer layout: the style width keeps applying.
    text = "We brought every single document for the meeting today, and nobody had asked for any of them."
    words = [Word(text=token, start=1 + position * .35, end=1.25 + position * .35, speaker_id="actor")
             for position, token in enumerate(text.split())]
    audio, fixture, config = (tmp_path / name for name in ("episode.wav", "words.json", "providers.yaml"))
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(2) * 16000 * 8)
    fixture.write_text(json.dumps({"words": [word.model_dump() for word in words]}), encoding="utf-8")
    config.write_text(yaml.safe_dump({"asr": {"fixture_path": str(fixture)}}), encoding="utf-8")

    result = generate_srt_from_audio(audio, tmp_path / "output.srt", tmp_path / "work", providers_path=config,
                                     no_llm=True, style_profile=StyleProfile(max_chars_per_line=26, min_cue_dur=.1))

    cues = parse_srt_text(result.output_srt.read_text(encoding="utf-8"))
    assert len(cues) > 1 and " ".join(cue.plain_text for cue in cues) == text
    assert all(len(cue.lines) <= 2 and all(display_width(line) <= 26 for line in cue.lines) for cue in cues)
    assert not [issue for issue in result.report["style_issues"] if issue["kind"] in {"line_length", "line_count"}]


_DELIVERED_RUNS = Path("work") / "upgrade-20261001" / "scratch" / "codex-validation" / "reprocessed-v10f"


def _delivered_stage_dirs() -> list[Path]:
    """Stage directories of the delivered-output replays, where this checkout has them."""
    roots = [Path(os.environ["DUBSYNC_DELIVERED_RUNS"])] if os.environ.get("DUBSYNC_DELIVERED_RUNS") else []
    roots.extend(parent / _DELIVERED_RUNS for parent in Path(__file__).resolve().parents)
    root = next((root for root in roots if root.is_dir()), None)
    if root is None:
        return []
    return sorted(path.parent for path in root.glob("*/stages/*/rebuild.json"))


_STAGES = _delivered_stage_dirs()


@pytest.mark.parametrize("stage", _STAGES or [pytest.param(None, marks=pytest.mark.skip(
    reason="the delivered-output evidence (work/upgrade-20261001/scratch) is not in this checkout"))],
    ids=lambda stage: stage.parent.parent.name[:24] if stage else "absent")
def test_delivered_customer_files_keep_every_unchanged_one_or_two_line_cue(stage):
    def load(name):
        return json.loads((stage / name).read_text(encoding="utf-8"))

    source = [Cue.model_validate(item) for item in load("ingest.json")["cues"]]
    profile = StyleProfile.model_validate(load("style_profile.json"))
    derived = derive_style_profile(source)
    assert (profile.max_chars_per_line, profile.max_lines_per_cue) == (
        derived.max_chars_per_line, derived.max_lines_per_cue)  # The default, source-derived style.
    rebuild = load("rebuild.json")
    pre_output = [Cue.model_validate(item) for item in rebuild["pre_output_cues"]]
    ownership = {int(cue_id): indices
                 for cue_id, indices in rebuild["pre_output_alignment"]["cue_word_indices"].items()}
    words = [Word.model_validate(item) for item in load("asr.json")["words"]]

    # The real final display pass, with every timed split available.
    segmented = split_crowded_output_cues(pre_output, words, ownership, profile, enforce_width=False)
    composed = compose_bracketed_annotations(segmented.cues, segmented.cue_word_indices, words=words,
                                             profile=profile, enforce_width=False)

    delivered = {cue.index: cue for cue in composed.cues}
    assert all(len(cue.text.splitlines()) <= 2 for cue in composed.cues)
    # Every spoken word is delivered once; screen text may repeat on consecutive displays.
    assert Counter(token for cue in composed.cues for token in alphanumeric_signature(
        speech_text_for_alignment(cue))) == Counter(
        token for cue in pre_output for token in alphanumeric_signature(speech_text_for_alignment(cue)))
    assert not Counter(token for cue in pre_output for token in alphanumeric_signature(cue.text)) - Counter(
        token for cue in composed.cues for token in alphanumeric_signature(cue.text))
    assert not segmented.flags and not segmented.expansions  # No cue here is over two lines before the pass.
    by_id = {cue.index: cue for cue in pre_output}
    hosts = set(composed.cue_annotations)
    checked = 0
    for cue in source:
        edited = by_id.get(cue.index)
        if len(cue.lines) > 2 or edited is None or edited.plain_text != cue.plain_text:
            continue
        if cue.index in composed.tracks or cue.index in composed.expansions:
            continue  # Screen text over speech, or speech divided to carry its pages.
        if cue.index in hosts:
            # A caption shares the display: the spoken line count, not its width, decides.
            if len(cue.lines) == 1:
                assert delivered[cue.index].lines[:1] == cue.lines
            continue
        assert delivered[cue.index].lines == cue.lines, (cue.index, cue.lines, delivered[cue.index].lines)
        checked += 1
    wide = [cue for cue in source if any(display_width(line) > profile.max_chars_per_line for line in cue.lines)]
    assert wide and checked > len(source) * .5
    # Speech is re-broken or divided only where a caption shares its display.
    assert all(hosts.intersection(children) for children in composed.expansions.values())
    assert all(hosts.intersection(flag.cue_ids) for flag in composed.flags if flag.kind in _LINE_LIMIT_KINDS)
