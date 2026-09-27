"""Run an uncached, paired MAI/Scribe benchmark and retain private evidence.

Example (load ELEVENLABS_API_KEY and OPENROUTER_API_KEY into the environment):
    python scripts/benchmark_transcription_models.py --manifest work/clips.json \
        --workdir work/transcription-comparison-20260905 --repetitions 3

Manifest paths are relative to the manifest, with this format:
    {"clips": [{"id": "dialogue", "audio": "../../Examples/dialogue test.wav",
                "reference_srt": "../../Examples/srt test.srt", "language": "de",
                "reference_note": "Source subtitles, not a verbatim gold transcript"}]}

Optional per-clip start_seconds/end_seconds crop the source and select overlapping
reference cues. Partial cue boundaries are counted and make text scores unreliable.
Only references explicitly marked human_verified_verbatim=true are described as
verified. Timing validity and speaker-label coverage do not measure timing accuracy
or diarization error rate. This script does not use DubSync's transcription cache.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Any, Callable, Sequence
import unicodedata
import wave

from dubsync.audio import probe_audio_duration
from dubsync.models import Word
from dubsync.srt_io import parse_srt_text
from dubsync.subtitle_annotations import text_without_bracketed_screen_text


SCRIBE_HOURLY_RATE = 0.22
MODELS = {"mai": "microsoft/mai-transcribe-2", "scribe": "scribe_v2"}
LIMITATIONS = [
    "Source subtitles may paraphrase, omit improvised speech, spell names differently, or be mistimed. "
    "WER/CER measure agreement with the supplied text, not proven recognition accuracy unless the "
    "reference was independently checked against the exact audio.",
    "Normalization: Unicode NFKC + casefold; remove subtitle tags, balanced square-bracket "
    "annotations, leading uppercase speaker labels, punctuation, and internal apostrophes. "
    "Numbers stay numeric, hyphens become spaces; CER excludes whitespace. "
    "Whitespace-based WER is unsuitable for languages without word spacing.",
    "Latency is wall-clock time for an uncached adapter call, including upload, HTTP/client setup, "
    "provider work, download, SDK retries, and any MAI chunking. Audio preparation is excluded. "
    "MAI and Scribe run sequentially with alternating order; small-sample p95 is descriptive only.",
    "Word timestamp validity and speaker-label coverage are structural checks. No human word-timing "
    "or speaker reference is available, so timestamp accuracy and diarization error rate are not measured.",
    "MAI costs use returned usage.cost in USD. Scribe costs are estimates at the configured hourly "
    "rate; subscription, discounts, taxes, billing minimums, and unreported failed-call charges are excluded.",
]


@dataclass(frozen=True)
class Clip:
    id: str
    audio: Path
    reference: Path
    language: str | None = None
    start_seconds: float = 0.0
    end_seconds: float | None = None
    human_verified_verbatim: bool = False
    reference_note: str = "Source SRT; not independently verified against the audio."


def normalize_text(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]*>|\{\\[^}]*\}", " ", text))
    text = text_without_bracketed_screen_text(text)
    # Uppercase labels only: normal dialogue containing a colon remains intact.
    text = re.sub(r"(?m)^\s*(?:[-–—]\s*)?[A-ZÄÖÜ][A-ZÄÖÜ0-9_ .-]{0,35}:\s*", "", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=\w)['’](?=\w)", "", text)
    text = "".join(char if char.isalnum() or unicodedata.category(char).startswith("M") else " " for char in text)
    return " ".join(text.split())


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int]:
    """Levenshtein counts with deterministic substitution/deletion/insertion tie order."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, token in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, other in enumerate(hypothesis, 1):
            if token == other:
                current.append(previous[j - 1])
                continue
            cost, substitutions, deletions, insertions = previous[j - 1]
            sub = (cost + 1, substitutions + 1, deletions, insertions)
            cost, substitutions, deletions, insertions = previous[j]
            delete = (cost + 1, substitutions, deletions + 1, insertions)
            cost, substitutions, deletions, insertions = current[j - 1]
            insert = (cost + 1, substitutions, deletions, insertions + 1)
            current.append(min((sub, delete, insert), key=lambda item: item[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {"errors": errors, "substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def error_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    words = edit_counts(ref.split(), hyp.split())
    ref_chars, hyp_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    # Only the scalar distance is needed for CER; C-backed rapidfuzz is much faster
    # than a Python character DP on a multi-minute transcript.
    from rapidfuzz.distance import Levenshtein

    character_errors = Levenshtein.distance(ref_chars, hyp_chars)
    return {
        "wer": words["errors"] / len(ref.split()) if ref else None,
        "cer": character_errors / len(ref_chars) if ref_chars else None,
        "word_errors": words["errors"], "character_errors": character_errors,
        "substitutions": words["substitutions"], "deletions": words["deletions"],
        "insertions": words["insertions"], "reference_words": len(ref.split()),
        "hypothesis_words": len(hyp.split()), "reference_characters": len(ref_chars),
    }


def word_statistics(words: Sequence[Word], duration_seconds: float) -> dict[str, Any]:
    invalid = sum(
        not (math.isfinite(word.start) and math.isfinite(word.end)
             and 0 <= word.start < word.end <= duration_seconds + 0.02)
        for word in words
    )
    labelled = [word for word in words if word.speaker_id is not None]
    return {
        "word_count": len(words), "invalid_timestamps": invalid,
        "valid_timestamp_ratio": (len(words) - invalid) / len(words) if words else None,
        "nonmonotonic_starts": sum(right.start < left.start for left, right in zip(words, words[1:])),
        "zero_duration_words": sum(word.start == word.end for word in words),
        "distinct_speakers": len({word.speaker_id for word in labelled}),
        "speaker_label_coverage": len(labelled) / len(words) if words else None,
    }


def load_manifest(path: Path) -> list[Clip]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = data.get("clips", []) if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a nonempty clips list.")
    result = []
    for item in entries:
        result.append(Clip(
            id=str(item["id"]), audio=(path.parent / item["audio"]).resolve(),
            reference=(path.parent / item["reference_srt"]).resolve(), language=item.get("language"),
            start_seconds=float(item.get("start_seconds", 0)),
            end_seconds=float(item["end_seconds"]) if item.get("end_seconds") is not None else None,
            human_verified_verbatim=item.get("human_verified_verbatim") is True,
            reference_note=str(item.get("reference_note", Clip.reference_note)),
        ))
    validate_clips(result)
    return result


def validate_clips(clips: Sequence[Clip]) -> None:
    if len({clip.id for clip in clips}) != len(clips):
        raise ValueError("Clip ids must be unique.")
    for clip in clips:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,70}", clip.id):
            raise ValueError("Clip ids must be simple filename-safe names.")
        if not math.isfinite(clip.start_seconds) or clip.start_seconds < 0:
            raise ValueError("Clip start must be finite and nonnegative.")
        if clip.end_seconds is not None and (not math.isfinite(clip.end_seconds) or clip.end_seconds <= clip.start_seconds):
            raise ValueError("Clip end must be finite and greater than start.")


def read_reference(path: Path, start_seconds: float, end_seconds: float) -> tuple[str, dict[str, Any]]:
    cues = parse_srt_text(path.read_text(encoding="utf-8-sig"))
    selected = [cue for cue in cues if cue.end_ms > start_seconds * 1000 and cue.start_ms < end_seconds * 1000]
    return "\n".join(cue.text for cue in selected), {
        "source_path": str(path.resolve()), "source_sha256": sha256_file(path),
        "selected_cues": len(selected),
        "partial_boundary_cues": sum(cue.start_ms < start_seconds * 1000 or cue.end_ms > end_seconds * 1000 for cue in selected),
        "human_verified_verbatim": False,
    }


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_clips(clips: Sequence[Clip], workdir: Path, ffmpeg: str = "ffmpeg") -> list[dict[str, Any]]:
    previous_path = workdir / "prepared.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))["clips"]
        if len(previous) != len(clips):
            raise ValueError("Existing workdir uses a different clip manifest; use a fresh workdir.")
        for clip, saved in zip(clips, previous):
            if (clip.id != saved["id"] or str(clip.audio) != saved["source_audio"]
                    or sha256_file(clip.audio) != saved["source_audio_sha256"]
                    or str(clip.reference) != saved["reference"]["source_path"]
                    or sha256_file(clip.reference) != saved["reference"]["source_sha256"]
                    or clip.start_seconds != saved["start_seconds"]
                    or clip.end_seconds != saved.get("requested_end_seconds")
                    or clip.language != saved.get("language")
                    or clip.human_verified_verbatim != saved["reference"]["human_verified_verbatim"]
                    or clip.reference_note != saved["reference"]["note"]
                    or sha256_file(Path(saved["normalized_audio"])) != saved["audio_sha256"]):
                raise ValueError("Existing benchmark evidence does not match current inputs; use a fresh workdir.")
        return previous
    prepared = []
    (workdir / "clips").mkdir(parents=True, exist_ok=True)
    for clip in clips:
        source_duration = probe_audio_duration(clip.audio)
        end = clip.end_seconds if clip.end_seconds is not None else source_duration
        if end > source_duration + 0.02 or clip.start_seconds >= source_duration:
            raise ValueError(f"Clip {clip.id} extends beyond its audio.")
        output = workdir / "clips" / f"{clip.id}.wav"
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                   "-ss", str(clip.start_seconds), "-i", str(clip.audio), "-t", str(end - clip.start_seconds),
                   "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)]
        subprocess.run(command, check=True, capture_output=True, timeout=1800)
        with wave.open(str(output), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
        reference_text, reference = read_reference(clip.reference, clip.start_seconds, end)
        reference.update(human_verified_verbatim=clip.human_verified_verbatim, note=clip.reference_note)
        prepared.append({
            "id": clip.id, "source_audio": str(clip.audio), "source_audio_sha256": sha256_file(clip.audio),
            "normalized_audio": str(output), "audio_sha256": sha256_file(output),
            "audio_format": "16000 Hz, mono, signed 16-bit PCM WAV", "duration_seconds": duration,
            "start_seconds": clip.start_seconds, "end_seconds": end,
            "requested_end_seconds": clip.end_seconds,
            "language": clip.language, "reference_text": reference_text,
            "normalized_reference": normalize_text(reference_text), "reference": reference,
        })
    return prepared


def redact_error(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    for name, value in os.environ.items():
        if (name.endswith("_API_KEY") or name.endswith("_TOKEN")) and len(value) >= 6:
            message = message.replace(value, "[REDACTED]")
    return re.sub(r"sk-or-[A-Za-z0-9_-]+", "[REDACTED]", message)[:1500]


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _percentile(values: Sequence[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(values) - 1) * proportion
    left, right = math.floor(index), math.ceil(index)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def summarize_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [run for run in runs if run["status"] == "ok"]
    latencies = [run["latency_seconds"] for run in successful]
    reference_words = sum(run["accuracy"]["reference_words"] for run in successful)
    reference_chars = sum(run["accuracy"]["reference_characters"] for run in successful)
    costs = [run["cost_usd"] for run in runs if run["cost_usd"] is not None]
    return {
        "successful_runs": len(successful), "failed_runs": len(runs) - len(successful),
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "median_real_time_factor": statistics.median(run["real_time_factor"] for run in successful) if successful else None,
        "pooled_wer": sum(run["accuracy"]["word_errors"] for run in successful) / reference_words if reference_words else None,
        "pooled_cer": sum(run["accuracy"]["character_errors"] for run in successful) / reference_chars if reference_chars else None,
        "cost_usd": sum(costs) if len(costs) == len(runs) else None,
        "known_cost_subtotal_usd": sum(costs), "cost_complete": len(costs) == len(runs),
        "cost_basis": runs[0]["cost_basis"] if runs else None,
    }


def run_benchmark(
    prepared: list[dict[str, Any]], workdir: Path, repetitions: int,
    adapter_factories: dict[str, Callable[[], Any]], scribe_hourly_rate: float = SCRIBE_HOURLY_RATE,
    *, force: bool = False,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("Repetitions must be at least one.")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "runs").mkdir(exist_ok=True)
    runs: list[dict[str, Any]] = []
    for clip_index, clip in enumerate(prepared):
        for repeat in range(repetitions):
            order = list(adapter_factories)
            if (clip_index + repeat) % 2:
                order.reverse()
            for model in order:
                output_path = workdir / "runs" / f"{clip['id']}-r{repeat + 1}-{model}.json"
                if output_path.exists() and not force:
                    saved = read_saved_run(output_path, clip, model, repeat + 1, scribe_hourly_rate)
                    runs.append(saved)
                    print(f"{clip['id']} repetition {repeat + 1}: {model} existing {saved['status']} reused; no API call", flush=True)
                    continue
                record: dict[str, Any] = {
                    "clip_id": clip["id"], "model": model, "model_id": MODELS.get(model, model),
                    "repetition": repeat + 1, "order_in_pair": order.index(model) + 1,
                    "audio_sha256": clip["audio_sha256"], "duration_seconds": clip["duration_seconds"],
                    "started_at_utc": datetime.now(timezone.utc).isoformat(), "cache_used": False,
                    "cost_usd": None,
                    "scribe_estimated_usd_per_hour": scribe_hourly_rate if model == "scribe" else None,
                    "cost_basis": "provider_reported_usage_cost" if model == "mai" else "estimate_at_configured_hourly_rate",
                }
                adapter = None
                started = time.perf_counter()
                try:
                    adapter = adapter_factories[model]()
                    # Set the same language hint for each provider; names and prompts are not supplied.
                    if clip.get("language"):
                        adapter.language_code = clip["language"]
                    words = adapter.transcribe(Path(clip["normalized_audio"]))
                    latency = time.perf_counter() - started
                    transcript = " ".join(word.text for word in words)
                    record.update(
                        status="ok", latency_seconds=latency,
                        real_time_factor=latency / clip["duration_seconds"],
                        transcript=transcript, normalized_transcript=normalize_text(transcript),
                        words=[word.model_dump() for word in words],
                        accuracy=error_metrics(clip["reference_text"], transcript),
                        word_statistics=word_statistics(words, clip["duration_seconds"]),
                    )
                    if model == "scribe":
                        record["cost_usd"] = clip["duration_seconds"] / 3600 * scribe_hourly_rate
                except Exception as error:
                    record.update(status="error", latency_seconds=time.perf_counter() - started, error=redact_error(error))
                usage = getattr(adapter, "last_usage", None)
                if isinstance(usage, dict):
                    # Keep only usage fields, never adapter state, request headers, or credentials.
                    record["usage"] = {key: usage[key] for key in ("cost", "seconds", "generation_ids", "request_count") if key in usage}
                    if model == "mai":
                        record["cost_usd"] = _number(usage.get("cost"))
                runs.append(record)
                write_json(output_path, record)
                print(f"{clip['id']} repetition {repeat + 1}: {model} {record['status']} ({record['latency_seconds']:.2f}s)", flush=True)
    # A model-only continuation still produces a combined report from matching
    # evidence already collected in this workdir.
    for model in MODELS.keys() - adapter_factories.keys():
        for clip in prepared:
            for repetition in range(1, repetitions + 1):
                path = workdir / "runs" / f"{clip['id']}-r{repetition}-{model}.json"
                if path.exists():
                    runs.append(read_saved_run(path, clip, model, repetition, scribe_hourly_rate))
    reported_models = [model for model in MODELS if any(run["model"] == model for run in runs)]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "repetitions": repetitions,
        "scribe_estimated_usd_per_hour": scribe_hourly_rate, "limitations": LIMITATIONS,
        "clips": prepared, "runs": runs,
        "models": {model: summarize_runs([run for run in runs if run["model"] == model]) for model in reported_models},
        "per_clip": {clip["id"]: {model: summarize_runs([run for run in runs if run["clip_id"] == clip["id"] and run["model"] == model])
                                  for model in reported_models} for clip in prepared},
    }


def read_saved_run(path: Path, clip: dict[str, Any], model: str, repetition: int, scribe_hourly_rate: float) -> dict[str, Any]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved.get("clip_id") != clip["id"] or saved.get("model") != model
            or saved.get("repetition") != repetition or saved.get("audio_sha256") != clip["audio_sha256"]
            or saved.get("duration_seconds") != clip["duration_seconds"]
            or saved.get("status") not in ("ok", "error")
            or (model == "scribe" and saved.get("scribe_estimated_usd_per_hour") != scribe_hourly_rate)):
        raise ValueError("Existing run does not match benchmark inputs; use a fresh workdir or --force.")
    return saved


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    def number(value: float | None, digits: int = 3) -> str:
        return f"{value:.{digits}f}" if value is not None else "unavailable"

    lines = ["# MAI-Transcribe 2 / Scribe v2 paired benchmark", "",
             f"Completed {result['created_at_utc']}; {result['repetitions']} repetitions per clip/model; uncached.", "",
             "WER/CER below are supplied-reference agreement unless the reference is marked independently verified.", "",
             "| Model | Runs OK / failed | Median latency (s) | p95 latency (s) | Median RTF | WER | CER | Total USD | Cost basis |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model, stats in result["models"].items():
        lines.append(f"| {MODELS.get(model, model)} | {stats['successful_runs']} / {stats['failed_runs']} | "
                     f"{number(stats['median_latency_seconds'])} | {number(stats['p95_latency_seconds'])} | "
                     f"{number(stats['median_real_time_factor'])} | {number(stats['pooled_wer'])} | "
                     f"{number(stats['pooled_cer'])} | {number(stats['cost_usd'], 6)} | {stats['cost_basis']} |")
    lines += ["", "## Clips and paired results", ""]
    for clip in result["clips"]:
        ref = clip["reference"]
        lines += [f"### {clip['id']}", "",
                  f"Audio: {clip['duration_seconds']:.3f}s; SHA-256 `{clip['audio_sha256']}`.",
                  f"Reference verified verbatim: {ref.get('human_verified_verbatim', False)}. "
                  f"Partial boundary cues: {ref.get('partial_boundary_cues', 0)}. {ref.get('note', '')}", "",
                  "| Model | Median latency (s) | Median RTF | WER | CER |",
                  "|---|---:|---:|---:|---:|"]
        for model, stats in result["per_clip"][clip["id"]].items():
            lines.append(f"| {model} | {number(stats['median_latency_seconds'])} | {number(stats['median_real_time_factor'])} | "
                         f"{number(stats['pooled_wer'])} | {number(stats['pooled_cer'])} |")
        lines.append("")
    lines += ["## Interpretation limits", ""] + [f"- {item}" for item in result["limitations"]]
    lines += ["", "Each call's raw word list, joined transcript, structural timestamp/diarization statistics, "
              "usage, and errors are retained in `runs/` and `benchmark.json`. Normalized WAVs are in `clips/`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_workdir(workdir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        relative = workdir.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("Workdir must be inside this repository's ignored work/ or workdir/ area.") from error
    check = subprocess.run(["git", "check-ignore", "--quiet", "--no-index", "--", str(relative / ".benchmark-private")], cwd=root, check=False)
    if check.returncode != 0:
        raise ValueError("Workdir must be git-ignored because it contains private audio and transcripts.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--audio", type=Path)
    parser.add_argument("--reference-srt", type=Path)
    parser.add_argument("--clip-id", default="sample")
    parser.add_argument("--language")
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--models", choices=list(MODELS), nargs="+", default=list(MODELS),
                        help="Run selected providers, e.g. --models scribe. Existing matching evidence is included in the report.")
    parser.add_argument("--force", action="store_true", help="Repeat selected paid API calls and replace their run evidence.")
    parser.add_argument("--scribe-hourly-rate", type=float, default=SCRIBE_HOURLY_RATE)
    parser.add_argument("--openrouter-key-file", type=Path, help="Extract one OpenRouter key silently; never copied to evidence.")
    parser.add_argument("--prepare-only", action="store_true", help="Normalize and record clips without API calls.")
    args = parser.parse_args(argv)
    try:
        if args.repetitions < 1 or not math.isfinite(args.scribe_hourly_rate) or args.scribe_hourly_rate < 0:
            raise ValueError("Repetitions must be positive and the estimated hourly rate finite/nonnegative.")
        if args.manifest:
            clips = load_manifest(args.manifest.resolve())
        elif args.reference_srt:
            clips = [Clip(args.clip_id, args.audio.resolve(), args.reference_srt.resolve(), args.language,
                          args.start_seconds, args.end_seconds)]
            validate_clips(clips)
        else:
            raise ValueError("--reference-srt is required with --audio.")
        workdir = args.workdir.resolve()
        validate_workdir(workdir)
        if args.openrouter_key_file:
            keys = set(re.findall(r"sk-or-(?:v1-)?[A-Za-z0-9_-]{20,}", args.openrouter_key_file.read_text(encoding="utf-8-sig")))
            if len(keys) != 1:
                raise ValueError("Key file must contain exactly one distinct OpenRouter API key.")
            os.environ["OPENROUTER_API_KEY"] = keys.pop()
        required_keys = {"mai": "OPENROUTER_API_KEY", "scribe": "ELEVENLABS_API_KEY"}
        if not args.prepare_only and not all(os.getenv(required_keys[model]) for model in args.models):
            raise ValueError("Selected models require their OPENROUTER_API_KEY / ELEVENLABS_API_KEY environment credentials.")
        prepared = prepare_clips(clips, workdir)
        write_json(workdir / "prepared.json", {"clips": prepared, "limitations": LIMITATIONS})
        if args.prepare_only:
            print(f"Prepared {len(prepared)} clips ({sum(clip['duration_seconds'] for clip in prepared):.3f}s); no API calls.")
            return 0
        from dubsync.mai_transcribe import MAITranscribeAdapter
        from dubsync.providers import ElevenLabsScribeAdapter

        factories = {"mai": MAITranscribeAdapter, "scribe": ElevenLabsScribeAdapter}
        result = run_benchmark(prepared, workdir, args.repetitions,
                               {model: factories[model] for model in dict.fromkeys(args.models)}, args.scribe_hourly_rate, force=args.force)
        write_json(workdir / "benchmark.json", result)
        write_markdown(workdir / "summary.md", result)
        print(f"Results: {workdir / 'summary.md'}")
        return 1 if any(run["status"] == "error" for run in result["runs"]) else 0
    except Exception as error:
        parser.exit(2, redact_error(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
