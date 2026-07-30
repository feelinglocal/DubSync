from __future__ import annotations

import json
import re

import pytest
import yaml
from typer.testing import CliRunner

import dubsync.profanity as german_profanity
from dubsync.aligner import align_cues_to_words
from dubsync.cli import app
from dubsync.models import Cue, QCFlag, Word
from dubsync.profanity import apply_german_profanity_censorship
from dubsync.srt_io import parse_srt_text
from dubsync.text_metrics import token_texts
from dubsync.tokenize import normalize_token
from dubsync.transcription import generate_srt_from_audio


def cue(index: int, text: str, *, start_ms: int = 0, end_ms: int = 1000) -> Cue:
    return Cue(index=index, start_ms=start_ms, end_ms=end_ms, lines=text.splitlines() or [text])


def test_preserves_existing_german_source_mask_when_output_uncensors_it():
    source = [cue(1, "Das ist verd*mmt knapp.")]
    rebuilt = [cue(1, "Das ist verdammt knapp.", start_ms=1200, end_ms=2400)]

    censored, flags = apply_german_profanity_censorship(rebuilt, source)

    assert censored[0].text == "Das ist verd*mmt knapp."
    assert censored[0].start_ms == 1200
    assert censored[0].end_ms == 2400
    assert flags[0].kind == "german_profanity_censored"
    assert flags[0].old_text == "verdammt"
    assert flags[0].new_text == "verd*mmt"


def test_censors_uncensored_german_profanity_from_source_and_output():
    source = [cue(1, "Mist, das ist verdammt schwer.")]
    rebuilt = [cue(1, "Mist, das ist verdammt schwer.")]

    censored, flags = apply_german_profanity_censorship(rebuilt, source)

    assert censored[0].text == "M*st, das ist verd*mmt schwer."
    assert [(flag.old_text, flag.new_text) for flag in flags] == [
        ("Mist", "M*st"),
        ("verdammt", "verd*mmt"),
    ]


def test_keeps_existing_masked_output_idempotently():
    source = [cue(1, "M*st, verd*mmt.")]
    rebuilt = [cue(1, "M*st, verd*mmt.")]

    first, first_flags = apply_german_profanity_censorship(rebuilt, source)
    second, second_flags = apply_german_profanity_censorship(first, source)

    assert first[0].text == "M*st, verd*mmt."
    assert first_flags == []
    assert second[0].text == first[0].text
    assert second_flags == []


def test_does_not_censor_german_substrings_inside_unlisted_words():
    source = [cue(1, "Der Misthaufen steht bei der Mistel.")]
    rebuilt = [cue(1, "Der Misthaufen steht bei der Mistel.")]

    censored, flags = apply_german_profanity_censorship(rebuilt, source)

    assert censored[0].text == "Der Misthaufen steht bei der Mistel."
    assert flags == []


def test_censors_case_and_common_german_compounds_without_changing_layout():
    source = [cue(1, "SCHEISSE!\nSchei\u00dfwetter.")]
    rebuilt = [cue(1, "SCHEISSE!\nSchei\u00dfwetter.")]

    censored, flags = apply_german_profanity_censorship(rebuilt, source)

    assert censored[0].lines == ["SCH*ISSE!", "Sch*i\u00dfwetter."]
    assert [(flag.old_text, flag.new_text) for flag in flags] == [
        ("SCHEISSE", "SCH*ISSE"),
        ("Schei\u00dfwetter", "Sch*i\u00dfwetter"),
    ]


def test_german_profanity_ruleset_has_a_stable_version_identifier():
    version = german_profanity.GERMAN_PROFANITY_RULESET_VERSION

    assert isinstance(version, str)
    assert re.fullmatch(r"de-[a-z0-9][a-z0-9._-]*", version)


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("verd*mmt", "verdammt"),
        ("Verd*mmte", "verdammt"),
        ("verd*mmter", "verdammt"),
        ("verdammt", "verdammt"),
        ("M*st", "mist"),
        ("Mist", "mist"),
        ("Sch*i\u00dfe", "scheisse"),
        ("SCHEISSE", "scheisse"),
        ("B*starde", "bastard"),
        ("Bastarde", "bastard"),
        ("Vollidiot", "idiot"),
        ("beschissene", "beschissen"),
    ],
)
def test_canonicalize_german_profanity_token_resolves_masks_inflections_and_compounds(
    value,
    canonical,
):
    assert german_profanity.canonicalize_german_profanity_token(value) == canonical


@pytest.mark.parametrize(
    "value",
    ["", "freundlich", "Optimist", "Marsch", "Marschall", "Mistel", "Mister", "Mistbeet"],
)
def test_canonicalize_german_profanity_token_rejects_non_profanity(value):
    assert german_profanity.canonicalize_german_profanity_token(value) is None


@pytest.mark.parametrize(
    ("uncensored", "censored"),
    [
        ("Verdammt", "Verd*mmt"),
        ("Verdammte", "Verd*mmte"),
        ("verdammter", "verd*mmter"),
        ("Mist", "M*st"),
        ("Schei\u00dfe", "Sch*i\u00dfe"),
        ("Bastarde", "B*starde"),
        ("schei\u00df", "sch*i\u00df"),
        ("Scheiss-Wetter", "Sch*iss-Wetter"),
        ("Idiot", "Id*ot"),
        ("Vollidiot", "Vollid*ot"),
        ("Idioten", "Id*oten"),
        ("Dummkopf", "D*mmkopf"),
        ("verarschen", "ver*rschen"),
        ("beschissen", "besch*ssen"),
        ("beschissene", "besch*ssene"),
        ("Arschloch", "*rschloch"),
        ("Fick", "F*ck"),
        ("ficken", "f*cken"),
        ("Fotze", "F*tze"),
        ("Hurensohn", "H*rensohn"),
        ("Wichser", "W*chser"),
        ("Kacke", "K*cke"),
        ("Pisser", "P*sser"),
        ("verpissen", "verp*ssen"),
        ("Depp", "D*pp"),
        ("Trottel", "Tr*ttel"),
        ("Bl\u00f6dmann", "Bl*dmann"),
    ],
)
def test_censor_german_profanity_text_uses_one_deterministic_mask_per_supported_term(
    uncensored,
    censored,
):
    result = german_profanity.censor_german_profanity_text(uncensored)

    assert result == censored
    assert result.count("*") == 1


def test_censor_german_profanity_text_is_idempotent_and_preserves_existing_masks():
    text = "Verd*mmt, M*st, Sch*i\u00dfe und B*starde!"

    assert german_profanity.censor_german_profanity_text(text) == text
    assert german_profanity.censor_german_profanity_text(
        german_profanity.censor_german_profanity_text(text)
    ) == text


def test_censor_german_profanity_text_preserves_punctuation_multiline_case_and_sharp_s():
    text = "SCHEISSE! Sch*i\u00dfe?\nBESCHISSENE Scheiss-Wetter."

    assert german_profanity.censor_german_profanity_text(text) == (
        "SCH*ISSE! Sch*i\u00dfe?\nBESCH*SSENE Sch*iss-Wetter."
    )


def test_censor_german_profanity_text_avoids_requested_false_positives():
    text = "Optimist Marsch Marschall Mistel Mister Mistbeet"

    assert german_profanity.censor_german_profanity_text(text) == text


@pytest.mark.parametrize(
    "safe_token",
    [
        "Fickleness",
        "Pissarro",
        "Kackelofen",
        "Wichserleben",
        "Fichte",
        "Wichtel",
        "Aschaffenburg",
        "Donaudampfschiff",
    ],
)
def test_profanity_guard_does_not_censor_safe_words_or_proper_nouns_with_shared_prefixes(
    safe_token,
):
    assert german_profanity.canonicalize_german_profanity_token(safe_token) is None
    assert german_profanity.censor_german_profanity_text(safe_token) == safe_token


def test_apply_uses_each_duplicate_source_mask_in_occurrence_order():
    source = [cue(1, "v*rdammt und verdam*t")]
    rebuilt = [cue(1, "verdammt und verdammt")]

    censored, _ = apply_german_profanity_censorship(rebuilt, source)

    assert censored[0].text == "v*rdammt und verdam*t"


@pytest.mark.parametrize(
    ("source_mask", "rebuilt_text"),
    [
        ("Sch***e", "Schei\u00dfe"),
        ("Sch**sse", "Scheisse"),
        ("B******e", "Bastarde"),
        ("v******t", "verdammt"),
        ("M**t", "Mist"),
        ("H*******n", "Hurensohn"),
    ],
)
def test_apply_restores_exact_source_mask_when_star_run_hides_multiple_letters(
    source_mask,
    rebuilt_text,
):
    source = [cue(1, f"{source_mask}!")]
    rebuilt = [cue(1, f"{rebuilt_text}!")]

    censored, flags = apply_german_profanity_censorship(rebuilt, source)

    assert censored[0].text == f"{source_mask}!"
    assert [(flag.old_text, flag.new_text) for flag in flags] == [
        (rebuilt_text, source_mask)
    ]


def test_apply_uses_global_source_mask_fallback_after_cue_move_or_reflow():
    source = [cue(1, "V*rdammt.")]
    rebuilt = [cue(9, "Verdammt,\nschon wieder.", start_ms=4200, end_ms=5900)]

    censored, _ = apply_german_profanity_censorship(rebuilt, source)

    assert censored[0].lines == ["V*rdammt,", "schon wieder."]


def test_apply_uses_canonical_mask_for_inserted_profanity_and_preserves_all_cue_metadata():
    source = [cue(1, "Geht weg.")]
    rebuilt = [
        Cue(
            index=7,
            start_ms=1234,
            end_ms=3456,
            lines=["Ihr Bastarde!", "Verschwindet."],
            speaker_id="speaker-b",
            character="KARL",
        )
    ]
    original_snapshot = rebuilt[0].model_dump(mode="json")

    censored, _ = apply_german_profanity_censorship(rebuilt, source)

    assert censored[0] is not rebuilt[0]
    assert rebuilt[0].model_dump(mode="json") == original_snapshot
    assert censored[0].model_dump(exclude={"lines"}) == rebuilt[0].model_dump(exclude={"lines"})
    assert censored[0].lines == ["Ihr B*starde!", "Verschwindet."]
    assert len(censored[0].lines) == len(rebuilt[0].lines)


def test_apply_accepts_no_source_cues_and_remains_idempotent():
    original = [cue(1, "Mist. Verdammt.")]

    first, first_flags = apply_german_profanity_censorship(original)
    second, second_flags = apply_german_profanity_censorship(first)

    assert [item.text for item in first] == ["M*st. Verd*mmt."]
    assert [item.text for item in second] == ["M*st. Verd*mmt."]
    assert original[0].text == "Mist. Verdammt."
    assert len(first_flags) == 2
    assert second_flags == []


def test_censor_german_profanity_flags_returns_sanitized_immutable_copies():
    source = [cue(1, "M*st. Sch*i\u00dfe. V*rdammt.")]
    flags = [
        QCFlag(
            kind="text_changed",
            cue_ids=[1],
            message="Mist / Schei\u00dfe / verdammt",
            severity="warning",
            confidence=0.91,
            old_text="Mist und Schei\u00dfe",
            new_text="verdammt",
            start=1.25,
            end=2.75,
        )
    ]
    original_snapshot = flags[0].model_dump(mode="json")

    sanitized = german_profanity.censor_german_profanity_flags(flags, source)

    assert sanitized is not flags
    assert sanitized[0] is not flags[0]
    assert flags[0].model_dump(mode="json") == original_snapshot
    assert sanitized[0].message == "M*st / Sch*i\u00dfe / V*rdammt"
    assert sanitized[0].old_text == "M*st und Sch*i\u00dfe"
    assert sanitized[0].new_text == "V*rdammt"
    assert sanitized[0].model_dump(exclude={"message", "old_text", "new_text"}) == flags[0].model_dump(
        exclude={"message", "old_text", "new_text"}
    )


def test_token_texts_keeps_supported_internal_asterisk_masks_as_single_tokens():
    assert token_texts("Verd*mmt, M*st! Sch*i\u00dfe. B*starde?") == [
        "Verd*mmt",
        "M*st",
        "Sch*i\u00dfe",
        "B*starde",
    ]


@pytest.mark.parametrize(
    ("masked", "uncensored"),
    [
        ("Verd*mmt", "Verdammt"),
        ("M*st", "Mist"),
        ("Sch*i\u00dfe", "Schei\u00dfe"),
        ("B*starde", "Bastarde"),
    ],
)
def test_normalize_token_treats_supported_masked_and_uncensored_forms_as_equal(masked, uncensored):
    assert normalize_token(masked) == normalize_token(uncensored)


def test_aligner_matches_masked_source_to_uncensored_asr_without_divergence():
    cues = [cue(1, "Verd*mmt, du Id*ot.", start_ms=0, end_ms=1200)]
    words = [
        Word(text="Verdammt", start=0.00, end=0.30, confidence=0.99),
        Word(text="du", start=0.32, end=0.45, confidence=0.99),
        Word(text="Idiot", start=0.47, end=0.80, confidence=0.99),
    ]

    result = align_cues_to_words(cues, words)

    assert result.anchor_coverage == 1.0
    assert result.divergence_spans == []
    assert result.unmatched_cue_ids == []
    assert result.cue_word_indices == {1: [0, 1, 2]}


def test_resume_verify_recensors_cached_uncensored_text_and_sanitizes_qc_flags(tmp_path):
    srt_path = tmp_path / "episode.srt"
    audio_path = tmp_path / "episode.wav"
    words_path = tmp_path / "episode.words.json"
    providers_path = tmp_path / "providers.yaml"
    workdir = tmp_path / "work"
    first_output = tmp_path / "first.srt"
    resumed_output = tmp_path / "resumed.srt"

    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nV*rdammt.\n\n",
        encoding="utf-8",
    )
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    words_path.write_text(
        json.dumps(
            {
                "words": [
                    {
                        "text": "Verdammt",
                        "start": 0.05,
                        "end": 0.65,
                        "confidence": 0.99,
                        "speaker_id": "A",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path.write_text(
        yaml.safe_dump({"asr": {"fixture_path": str(words_path)}}),
        encoding="utf-8",
    )

    first = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(first_output),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
            "--no-llm",
        ],
    )
    assert first.exit_code == 0, first.output

    rebuild_path = workdir / "episode" / "rebuild.json"
    rebuild = json.loads(rebuild_path.read_text(encoding="utf-8"))
    rebuild["cues"][0]["lines"] = ["Verdammt."]
    rebuild_path.write_text(json.dumps(rebuild), encoding="utf-8")

    resumed = CliRunner().invoke(
        app,
        [
            "sync",
            str(srt_path),
            str(audio_path),
            "-o",
            str(resumed_output),
            "--providers",
            str(providers_path),
            "--workdir",
            str(workdir),
            "--resume",
            "verify",
            "--no-llm",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    resumed_cues = parse_srt_text(resumed_output.read_text(encoding="utf-8"))
    assert resumed_cues[0].text == "V*rdammt."
    report_text = (workdir / "episode" / "qc_report.json").read_text(encoding="utf-8")
    assert "Verdammt" not in report_text


def test_audio_only_generation_censors_asr_profanity_in_all_public_artifacts(tmp_path):
    audio_path = tmp_path / "dialogue.wav"
    audio_path.write_bytes(b"fixture audio")
    words_path = tmp_path / "dialogue.words.json"
    words_path.write_text(
        json.dumps(
            {
                "words": [
                    {
                        "text": "Mist.",
                        "start": 0.10,
                        "end": 0.60,
                        "confidence": 0.99,
                        "speaker_id": "A",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    providers_path = tmp_path / "providers.yaml"
    providers_path.write_text(
        yaml.safe_dump({"asr": {"fixture_path": str(words_path)}}),
        encoding="utf-8",
    )
    output_path = tmp_path / "dialogue.generated.srt"

    result = generate_srt_from_audio(
        audio_path=audio_path,
        output_path=output_path,
        workdir=tmp_path / "work",
        providers_path=providers_path,
        no_llm=True,
    )

    assert parse_srt_text(output_path.read_text(encoding="utf-8"))[0].text == "M*st."
    generate_text = (result.episode_workdir / "generate.json").read_text(encoding="utf-8")
    report_text = (result.episode_workdir / "qc_report.json").read_text(encoding="utf-8")
    assert '"M*st."' in generate_text
    assert "Mist" not in generate_text
    assert "Mist" not in report_text
