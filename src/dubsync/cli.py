from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import click
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .config import write_style_profile
from .evaluation import evaluate_against_golden
from .models import Cue, QCFlag
from .pipeline import sync_episode
from .srt_io import SRTParseError, parse_srt_text
from .style_profile import derive_style_profile
from .transcription import generate_srt_from_audio

app = typer.Typer(no_args_is_help=True, help="Synchronize dubbed VO audio with customer SRT files.")
console = Console()
RESUME_STAGE_HELP = "Resume from asr, align, adjudicate, rebuild, or verify."
RESUME_STAGES = {"asr", "align", "adjudicate", "rebuild", "verify"}
GENERATED_SRT_NAMES = {"changes.diff.srt"}
GENERATED_SRT_SUFFIXES = (".synced.srt", ".changes.diff.srt")


@app.command()
def sync(
    srt: Path = typer.Argument(..., exists=True, readable=True, help="Customer-supplied SRT file."),
    audio: Path = typer.Argument(..., exists=True, readable=True, help="VO-only dubbed WAV/MP3 audio."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Synced SRT output path."),
    style: Optional[Path] = typer.Option(None, "--style", exists=True, readable=True, help="Style profile YAML."),
    providers: Optional[Path] = typer.Option(None, "--providers", exists=True, readable=True, help="Provider config YAML."),
    workdir: Path = typer.Option(Path("workdir"), "--workdir", help="Stage artifact directory."),
    local: bool = typer.Option(False, "--local", help="Prefer local ASR providers where configured."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Timing-only mode with full QC."),
    fps: Optional[float] = typer.Option(None, "--fps", help="Override detected frame rate."),
    resume: Optional[str] = typer.Option(None, "--resume", help=RESUME_STAGE_HELP),
) -> None:
    _load_dotenv()
    output_path = output or srt.with_name(f"{srt.stem}.synced.srt")
    result = _sync_episode_or_exit(
        srt,
        audio,
        output_path,
        workdir,
        style,
        providers,
        no_llm=no_llm,
        fps=fps,
        resume=_validate_resume_stage(resume),
        local=local,
    )
    console.print(f"[green]Wrote[/green] {result.output_srt}")
    console.print(f"[green]Artifacts[/green] {result.episode_workdir}")
    console.print("Cost meter")
    console.print(result.cost_meter.to_json())


@app.command()
def batch(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    style: Optional[Path] = typer.Option(None, "--style", exists=True, readable=True),
    providers: Optional[Path] = typer.Option(None, "--providers", exists=True, readable=True),
    workdir: Path = typer.Option(Path("workdir"), "--workdir"),
    local: bool = typer.Option(False, "--local", help="Prefer local ASR providers where configured."),
    no_llm: bool = typer.Option(False, "--no-llm"),
    fps: Optional[float] = typer.Option(None, "--fps", help="Override detected frame rate."),
    resume: Optional[str] = typer.Option(None, "--resume", help=RESUME_STAGE_HELP),
) -> None:
    _load_dotenv()
    resume_stage = _validate_resume_stage(resume)
    srt_files = _episode_srt_files(folder)
    if not srt_files:
        raise typer.BadParameter("folder contains no source .srt files")
    processed = 0
    for srt_path in srt_files:
        audio = _matching_audio(srt_path)
        if audio is None:
            console.print(f"[yellow]Skipping[/yellow] {srt_path}: no matching WAV/MP3")
            continue
        output = srt_path.with_name(f"{srt_path.stem}.synced.srt")
        result = _sync_episode_or_exit(srt_path, audio, output, workdir, style, providers, no_llm=no_llm, fps=fps, resume=resume_stage, local=local)
        console.print(f"[green]Wrote[/green] {result.output_srt}")
        console.print(f"[green]Artifacts[/green] {result.episode_workdir}")
        console.print("Cost meter")
        console.print(result.cost_meter.to_json())
        processed += 1
    if processed == 0:
        raise typer.BadParameter("no episodes processed; no source SRT had a matching WAV/MP3")


@app.command()
def generate(
    audio: Path = typer.Argument(..., exists=True, readable=True, help="Dialogue audio to transcribe into SRT."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Generated SRT output path."),
    style: Optional[Path] = typer.Option(None, "--style", exists=True, readable=True, help="Style profile YAML."),
    providers: Optional[Path] = typer.Option(None, "--providers", exists=True, readable=True, help="Provider config YAML."),
    workdir: Path = typer.Option(Path("workdir"), "--workdir", help="Stage artifact directory."),
    local: bool = typer.Option(False, "--local", help="Use local ASR and disable cloud language passes."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the punctuation language pass."),
    fps: Optional[float] = typer.Option(None, "--fps", help="Output frame rate."),
) -> None:
    _load_dotenv()
    output_path = output or audio.with_name(f"{audio.stem}.generated.srt")
    try:
        result = generate_srt_from_audio(
            audio,
            output_path,
            workdir,
            style_path=style,
            providers_path=providers,
            no_llm=no_llm,
            fps=fps,
            local=local,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[green]Wrote[/green] {result.output_srt}")
    console.print(f"[green]Artifacts[/green] {result.episode_workdir}")
    console.print("Cost meter")
    console.print(result.cost_meter.to_json())


@app.command()
def profile(
    sample_srt: Path = typer.Argument(..., exists=True, readable=True, help="Sample SRT that defines house style."),
    output: Path = typer.Option(Path("style_profile.yaml"), "-o", "--output", help="Output style profile YAML."),
) -> None:
    cues = _parse_profile_srt(sample_srt)
    style_profile = derive_style_profile(cues)
    write_style_profile(output, style_profile)
    table = Table(title="Derived Style Profile")
    table.add_column("Property")
    table.add_column("Value")
    for key, value in style_profile.model_dump(exclude_none=True).items():
        table.add_row(key, json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value))
    console.print(table)
    console.print(f"[green]Wrote[/green] {output}")


@app.command()
def report(
    workdir: Path = typer.Argument(..., exists=True, readable=True),
    synced: Optional[Path] = typer.Option(None, "--synced", exists=True, readable=True, help="Synced SRT to compare against a golden SRT."),
    golden: Optional[Path] = typer.Option(None, "--golden", exists=True, readable=True, help="Golden human-synced SRT for evaluation metrics."),
    fps: float = typer.Option(30.0, "--fps", help="Frame rate for timing tolerance metrics."),
) -> None:
    report_path = workdir / "qc_report.json"
    if not report_path.exists():
        candidates = sorted(workdir.glob("*/qc_report.json"))
        if not candidates:
            raise typer.BadParameter("no qc_report.json found")
        if len(candidates) > 1:
            raise typer.BadParameter("multiple qc_report.json files found; pass a specific episode workdir")
        report_path = candidates[0]
    payload = _load_report_payload(report_path)
    if synced is not None or golden is not None:
        if synced is None or golden is None:
            raise typer.BadParameter("--synced and --golden must be provided together")
        predicted_cues = _parse_report_srt(synced, "--synced")
        golden_cues = _parse_report_srt(golden, "--golden")
        raw_flags = payload.get("flags", [])
        flags = [QCFlag.model_validate(flag) for flag in raw_flags] if isinstance(raw_flags, list) else []
        style_violations = int(payload.get("summary", {}).get("style_violations", 0)) if isinstance(payload.get("summary"), dict) else 0
        source_cues = _load_report_source_cues(report_path.parent / "ingest.json")
        payload["evaluation"] = evaluate_against_golden(
            predicted_cues,
            golden_cues,
            fps=fps,
            flags=flags,
            style_violations=style_violations,
            source=source_cues,
        )
    console.print(json.dumps(payload, indent=2, ensure_ascii=False))


def _matching_audio(srt_path: Path) -> Path | None:
    for suffix in (".wav", ".mp3"):
        candidate = srt_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _episode_srt_files(folder: Path) -> list[Path]:
    return [path for path in sorted(folder.glob("*.srt")) if not _is_generated_srt_artifact(path)]


def _is_generated_srt_artifact(path: Path) -> bool:
    name = path.name.lower()
    return name in GENERATED_SRT_NAMES or any(name.endswith(suffix) for suffix in GENERATED_SRT_SUFFIXES)


def _sync_episode_or_exit(*args, **kwargs):
    try:
        return sync_episode(*args, **kwargs)
    except (RuntimeError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


def _load_dotenv() -> None:
    env_path = Path.cwd() / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_dotenv_fallback(env_path)
        return
    load_dotenv(dotenv_path=env_path, override=False)


def _load_dotenv_fallback(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")


def _load_report_payload(report_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise click.ClickException(f"invalid qc_report.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("invalid qc_report.json: expected a JSON object")
    return payload


def _parse_profile_srt(path: Path):
    try:
        return parse_srt_text(path.read_text(encoding="utf-8-sig"))
    except (SRTParseError, OSError) as exc:
        raise click.ClickException(f"invalid sample SRT: {exc}") from exc


def _parse_report_srt(path: Path, option_name: str):
    try:
        return parse_srt_text(path.read_text(encoding="utf-8-sig"))
    except (SRTParseError, OSError) as exc:
        raise click.ClickException(f"invalid {option_name} SRT: {exc}") from exc


def _load_report_source_cues(path: Path) -> list[Cue] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise click.ClickException(f"invalid ingest.json: {exc}") from exc
    raw_cues = payload.get("cues") if isinstance(payload, dict) else None
    if not isinstance(raw_cues, list):
        raise click.ClickException("invalid ingest.json: expected a cues list")
    try:
        return [Cue.model_validate(item) for item in raw_cues]
    except ValidationError as exc:
        first_error = exc.errors()[0]
        field = ".".join(str(part) for part in first_error.get("loc", ())) or "cue"
        message = str(first_error.get("msg", "invalid cue"))
        raise click.ClickException(f"invalid ingest.json: {field}: {message}") from exc


def _validate_resume_stage(resume: str | None) -> str | None:
    if resume is None:
        return None
    stage = resume.strip().lower()
    if stage not in RESUME_STAGES:
        raise typer.BadParameter(f"Unsupported resume stage: {resume}. Use one of: {', '.join(sorted(RESUME_STAGES))}.")
    return stage
