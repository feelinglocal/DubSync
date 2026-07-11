from __future__ import annotations

import subprocess
from pathlib import Path


class AudioNormalizeError(RuntimeError):
    pass


def normalize_audio(source: Path, dest: Path, ffmpeg: str = "ffmpeg") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioNormalizeError("ffmpeg was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise AudioNormalizeError(f"ffmpeg failed: {message}") from exc
    return dest
