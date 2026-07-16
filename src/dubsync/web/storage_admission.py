from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException

from ..audio import AudioNormalizeError, predicted_normalized_audio_bytes
from ..config import load_yaml
from .intake_guard import STORAGE_CAPACITY_DETAIL
from .jobs import JobStorageLimitError, JobStore, Processor, default_processor
from .settings import WebSettings


def requires_audio_normalization(settings: WebSettings, processor: Processor) -> bool:
    if processor is not default_processor:
        return False
    provider_config = load_yaml(settings.providers_path)
    asr_config = provider_config.get("asr", {}) if isinstance(provider_config, dict) else {}
    return not (isinstance(asr_config, dict) and asr_config.get("fixture_path"))


async def reserve_processing_storage(
    *,
    directory: Path,
    audio_path: Path,
    settings: WebSettings,
    store: JobStore,
    normalization_required: bool,
    audio_duration_probe: Callable[..., float],
) -> None:
    additional_bytes = settings.max_job_work_bytes
    if normalization_required:
        try:
            duration = await asyncio.to_thread(
                audio_duration_probe,
                audio_path,
                timeout_seconds=settings.ffprobe_timeout_seconds,
            )
        except (AudioNormalizeError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Audio duration could not be verified.") from exc
        if duration > settings.max_audio_duration_seconds:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Audio is too long. "
                    f"Use audio no longer than {settings.max_audio_duration_seconds:g} seconds."
                ),
            )
        predicted_bytes = predicted_normalized_audio_bytes(duration)
        if predicted_bytes > settings.max_normalized_audio_bytes:
            raise HTTPException(
                status_code=413,
                detail="Audio requires too much normalized processing storage.",
            )
        additional_bytes += predicted_bytes

    try:
        reservation = store.reserve_job_storage(
            directory,
            additional_bytes=additional_bytes,
            max_job_storage_bytes=settings.max_job_storage_bytes,
        )
        committed = store.storage_usage_bytes()
        actual = store.job_storage_bytes(directory)
    except (JobStorageLimitError, OSError) as exc:
        raise HTTPException(
            status_code=413,
            detail="This job requires too much processing storage.",
        ) from exc
    if committed > settings.max_retained_storage_bytes:
        raise HTTPException(status_code=507, detail=STORAGE_CAPACITY_DETAIL)

    try:
        free_bytes = shutil.disk_usage(settings.data_dir).free
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Storage capacity could not be verified.") from exc
    unmaterialized_bytes = max(0, reservation - actual)
    if free_bytes < unmaterialized_bytes + settings.min_free_storage_bytes:
        raise HTTPException(status_code=507, detail=STORAGE_CAPACITY_DETAIL)
