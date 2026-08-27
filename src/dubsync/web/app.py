from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Annotated, BinaryIO
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import FormData, UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..audio import probe_audio_duration
from ..providers import (
    GEMINI_TRANSCRIBE_DISABLED_MESSAGE,
    GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS,
    GEMINI_TRANSCRIBE_MODEL,
)
from .batch_uploads import (
    AUDIO_EXTENSIONS,
    MAX_BATCH_ITEMS,
    batch_upload_plan as _batch_upload_plan,
    validate_audio as _validate_audio,
    validate_source_filename as _validate_source_filename,
    validate_subtitle as _validate_subtitle,
)
from .generation_styles import (
    GenerationStyleError,
    SYNC_MAX_LINES_MAX,
    SYNC_MAX_LINES_MIN,
    parse_generation_style_request,
    public_generation_styles,
    public_sync_style_limits,
    resolve_generation_style,
    sync_style_from_max_lines,
)
from .jobs import (
    JobMode,
    JobRecord,
    JobService,
    OutstandingJobLimitError,
    Processor,
    default_processor,
    new_job_record,
)
from .intake_guard import (
    BATCH_BODY_LIMIT_DETAIL,
    SINGLE_BODY_LIMIT_DETAIL,
    SecurityAndIntakeMiddleware,
)
from .security import SlidingWindowRateLimiter, hash_job_token, valid_job_token
from .settings import WebSettings
from .srt_uploads import read_validated_srt_upload
from .storage_admission import reserve_processing_storage, requires_audio_normalization

logger = logging.getLogger(__name__)

FPS_VALUES = {23.976, 24.0, 25.0, 29.97, 30.0}
LANGUAGE_RE = re.compile(r"^(auto|[A-Za-z]{2,8}(?:-[A-Za-z]{2,8})?)$")
PUBLIC_ROOT_FILES = frozenset({"favicon.svg", "robots.txt", "site.webmanifest", "sitemap.xml", "theme-init.js"})
FRONTEND_ROUTE_METADATA = {
    "terms": (
        "Terms of Service | DubSync",
        "Terms for using DubSync subtitle synchronization and audio-to-SRT processing.",
    ),
    "privacy": (
        "Privacy Policy | DubSync",
        "How DubSync processes, protects, transfers, and deletes subtitle job data.",
    ),
    "payments": (
        "Payments and Refunds | DubSync",
        "Manual billing, tax handling, cancellations, reruns, and refund eligibility for DubSync jobs.",
    ),
}
SITE_ORIGIN = "https://dubsync.onrender.com"
MAX_BATCH_PARSER_FILES = 21
MAX_BATCH_PARSER_FIELDS = 6
MAX_BATCH_FIELD_BYTES = 64 * 1024
MAX_BATCH_DOWNLOAD_BODY_BYTES = 16 * 1024
BATCH_ARCHIVE_SPOOL_BYTES = 8 * 1024 * 1024
ARCHIVE_COPY_CHUNK_BYTES = 1024 * 1024
MAX_SINGLE_PARSER_FILES = 3
MAX_SINGLE_PARSER_FIELDS = 6
MAX_QC_RESULT_METADATA_BYTES = 16 * 1024 * 1024
FPS_RESULT_SOURCES = frozenset({"detected", "fallback", "explicit"})
DEFAULT_TRANSCRIPTION_PROVIDER = "default"
TRANSCRIPTION_PROVIDER_VALUES = frozenset({DEFAULT_TRANSCRIPTION_PROVIDER, GEMINI_TRANSCRIBE_MODEL})


def create_app(
    *,
    settings: WebSettings | None = None,
    processor: Processor = default_processor,
    audio_duration_probe: Callable[..., float] = probe_audio_duration,
) -> FastAPI:
    resolved_settings = settings or WebSettings.from_env()
    resolved_settings.ensure_directories()
    service = JobService(resolved_settings, processor)
    limiter = SlidingWindowRateLimiter(resolved_settings.max_submissions_per_hour)
    normalization_required = requires_audio_normalization(resolved_settings, processor)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.start()
        yield
        service.shutdown()

    app = FastAPI(
        title="DubSync",
        version="0.2.0",
        docs_url="/api/docs" if os.getenv("DUBSYNC_ENABLE_DOCS") == "1" else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.jobs = service
    app.add_middleware(
        SecurityAndIntakeMiddleware,
        settings=resolved_settings,
        limiter=limiter,
        storage_usage_bytes=service.store.storage_usage_bytes,
    )

    def _rollback_intake(job_ids: list[str], directories: list[Path]) -> None:
        try:
            service.store.delete_many(job_ids)
        except Exception:
            logger.exception("Could not remove persisted jobs after intake failed")
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)

    @app.exception_handler(StarletteHTTPException)
    async def body_limit_exception_handler(request: Request, exc: StarletteHTTPException):
        if exc.detail in {SINGLE_BODY_LIMIT_DETAIL, BATCH_BODY_LIMIT_DETAIL}:
            exc = StarletteHTTPException(
                status_code=413,
                detail=exc.detail,
                headers=exc.headers,
            )
        return await http_exception_handler(request, exc)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        if not service.store.healthcheck():
            raise HTTPException(status_code=503, detail="Storage is unavailable.")
        payload: dict[str, object] = {"status": "ok", "service": "dubsync", "version": app.version}
        if commit := os.getenv("RENDER_GIT_COMMIT"):
            payload["commit"] = commit
        return payload

    @app.get("/api/config")
    def public_config() -> dict[str, object]:
        access_code_required = resolved_settings.require_job_access_code or bool(resolved_settings.job_access_code)
        jobs_available = not resolved_settings.require_job_access_code or bool(resolved_settings.job_access_code)
        return {
            "retention_hours": resolved_settings.retention_hours,
            "max_upload_bytes": resolved_settings.max_upload_bytes,
            "max_srt_bytes": resolved_settings.max_srt_bytes,
            "max_batch_upload_bytes": resolved_settings.max_batch_upload_bytes,
            "max_batch_files": MAX_BATCH_ITEMS,
            "audio_extensions": sorted(AUDIO_EXTENSIONS),
            "fps_values": sorted(FPS_VALUES),
            "pricing": {
                "generate": {"usd_per_minute": 0.40, "minimum_usd": 20.0},
                "sync": {"usd_per_minute": 0.60, "minimum_usd": 30.0},
                "precision": {"usd_per_minute": 0.90, "minimum_usd": 50.0},
            },
            "billing_enabled": False,
            "access_code_required": access_code_required,
            "jobs_available": jobs_available,
            "gemini_transcribe_testing_available": _gemini_transcribe_web_available(resolved_settings),
            "gemini_transcribe_max_audio_seconds": int(GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS),
            "generation_styles": public_generation_styles(),
            "sync_style_limits": public_sync_style_limits(),
        }

    async def _create_single_job(
        *,
        mode: str,
        audio: StarletteUploadFile,
        subtitle: StarletteUploadFile | None,
        style_sample: StarletteUploadFile | None,
        fps: float | None,
        language: str,
        style: str,
        sync_max_lines_per_cue: int | None,
        transcription_provider: str,
    ) -> dict[str, object]:
        normalized_mode = _validate_mode(mode)
        if normalized_mode == "sync" and subtitle is None:
            raise HTTPException(status_code=422, detail="An original SRT is required for sync mode.")
        _validate_options(mode=normalized_mode, fps=fps, language=language)
        selected_transcription_provider = _validate_transcription_provider(
            transcription_provider,
            settings=resolved_settings,
        )
        resolved_sync_style = _sync_style_for_mode(normalized_mode, sync_max_lines_per_cue)
        source_name = _validate_source_filename(audio)
        audio_extension = _validate_audio(audio)
        subtitle_bytes: bytes | None = None
        if subtitle is not None:
            _validate_subtitle(subtitle)
            validated_subtitle = await read_validated_srt_upload(
                subtitle,
                max_bytes=resolved_settings.max_srt_bytes,
                max_line_bytes=resolved_settings.max_srt_line_bytes,
                parse_limits=resolved_settings.srt_parse_limits,
                label="SRT",
            )
            subtitle_bytes = validated_subtitle.data

        resolved_style = "source"
        style_sample_bytes: bytes | None = None
        if normalized_mode == "generate":
            generation_fps = _require_generation_fps(fps)
            try:
                style_request = parse_generation_style_request(style)
            except GenerationStyleError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            sample_cues = None
            if style_request.source == "sample":
                if style_sample is None:
                    raise HTTPException(status_code=422, detail="An SRT style example is required.")
                _validate_subtitle(style_sample)
                validated_style = await read_validated_srt_upload(
                    style_sample,
                    max_bytes=resolved_settings.max_srt_bytes,
                    max_line_bytes=resolved_settings.max_srt_line_bytes,
                    parse_limits=resolved_settings.srt_parse_limits,
                    label="SRT style example",
                )
                style_sample_bytes = validated_style.data
                sample_cues = list(validated_style.cues)
            elif style_sample is not None:
                await style_sample.close()
            try:
                resolved_style = resolve_generation_style(
                    style_request,
                    fps=generation_fps,
                    sample_cues=sample_cues,
                ).model_dump_json()
            except GenerationStyleError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        elif style_sample is not None:
            await style_sample.close()
        if normalized_mode == "sync":
            resolved_style = resolved_sync_style

        service.store.delete_expired()
        job_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        directory = (resolved_settings.data_dir / f"job-{job_id}").resolve()
        directory.mkdir(parents=False, exist_ok=False)
        audio_path = directory / f"audio{audio_extension}"
        subtitle_path = directory / "original.srt" if subtitle is not None else None
        succeeded = False
        try:
            await _save_upload(audio, audio_path, resolved_settings.max_upload_bytes)
            if subtitle_bytes is not None and subtitle_path is not None:
                subtitle_path.write_bytes(subtitle_bytes)
            if style_sample_bytes is not None:
                (directory / "style-example.srt").write_bytes(style_sample_bytes)
            await reserve_processing_storage(
                directory=directory,
                audio_path=audio_path,
                settings=resolved_settings,
                store=service.store,
                normalization_required=(
                    normalization_required
                    or selected_transcription_provider == GEMINI_TRANSCRIBE_MODEL
                ),
                audio_duration_probe=audio_duration_probe,
                max_audio_duration_seconds=(
                    GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS
                    if selected_transcription_provider == GEMINI_TRANSCRIBE_MODEL
                    else None
                ),
            )
            job = new_job_record(
                job_id=job_id,
                token_hash=hash_job_token(token),
                mode=normalized_mode,
                directory=directory,
                audio_path=audio_path,
                srt_path=subtitle_path,
                fps=fps,
                language=language.lower(),
                style=resolved_style,
                retention_hours=resolved_settings.retention_hours,
                source_name=source_name,
                transcription_provider=selected_transcription_provider,
            )
            try:
                service.store.create_many(
                    [job],
                    max_outstanding=resolved_settings.max_outstanding_child_jobs,
                )
            except OutstandingJobLimitError as exc:
                raise HTTPException(
                    status_code=429,
                    detail="Too many files are already queued. Wait for current jobs to finish.",
                ) from exc
            service.submit(job)
            latest = service.store.get(job_id)
            if latest is None:
                raise HTTPException(status_code=500, detail="Could not create job.")
            response = _public_job(latest, token=token)
            succeeded = True
            return response
        finally:
            if not succeeded:
                _rollback_intake([job_id], [directory])

    @app.post("/api/jobs", status_code=202)
    async def create_job(request: Request) -> dict[str, object]:
        form = await _parse_single_form(request)
        try:
            _validate_single_form_shape(form)
            audio_uploads = _batch_file_field(form, "audio")
            subtitle_uploads = _batch_file_field(form, "subtitle")
            style_samples = _batch_file_field(form, "style_sample")
            if len(audio_uploads) != 1:
                raise HTTPException(status_code=422, detail="Exactly one audio file is required.")
            if len(subtitle_uploads) > 1:
                raise HTTPException(status_code=422, detail="Only one original SRT is allowed.")
            if len(style_samples) > 1:
                raise HTTPException(status_code=422, detail="Only one SRT style example is allowed.")
            normalized_mode = _validate_mode(_batch_text_field(form, "mode"))
            return await _create_single_job(
                mode=normalized_mode,
                audio=audio_uploads[0],
                subtitle=subtitle_uploads[0] if subtitle_uploads else None,
                style_sample=style_samples[0] if style_samples else None,
                fps=_batch_fps_field(form, mode=normalized_mode),
                language=_batch_text_field(form, "language", default="auto"),
                style=_batch_text_field(form, "style", default="standard"),
                sync_max_lines_per_cue=_batch_sync_max_lines_field(form),
                transcription_provider=_batch_text_field(
                    form,
                    "transcription_provider",
                    default=DEFAULT_TRANSCRIPTION_PROVIDER,
                ),
            )
        finally:
            await form.close()

    @app.post("/api/batches", status_code=202)
    async def create_batch(request: Request) -> dict[str, object]:
        form = await _parse_batch_form(request)
        try:
            _validate_batch_form_shape(form)
            normalized_mode = _validate_mode(_batch_text_field(form, "mode"))
            fps = _batch_fps_field(form, mode=normalized_mode)
            language = _batch_text_field(form, "language", default="auto")
            style = _batch_text_field(form, "style", default="standard")
            sync_max_lines_per_cue = _batch_sync_max_lines_field(form)
            _validate_options(mode=normalized_mode, fps=fps, language=language)
            selected_transcription_provider = _validate_transcription_provider(
                _batch_text_field(form, "transcription_provider", default=DEFAULT_TRANSCRIPTION_PROVIDER),
                settings=resolved_settings,
            )
            resolved_sync_style = _sync_style_for_mode(normalized_mode, sync_max_lines_per_cue)

            audio_uploads = _batch_file_field(form, "audio")
            subtitle_uploads = _batch_file_field(form, "subtitle")
            style_samples = _batch_file_field(form, "style_sample")
            if len(style_samples) > 1:
                raise HTTPException(status_code=422, detail="Only one SRT style example is allowed.")
            style_sample = style_samples[0] if style_samples else None
            plan = _batch_upload_plan(normalized_mode, audio_uploads, subtitle_uploads)

            sized_uploads = [
                *((upload, resolved_settings.max_upload_bytes) for _, upload, _, _ in plan),
                *((upload, resolved_settings.max_srt_bytes) for _, _, _, upload in plan if upload is not None),
                *(
                    ((style_sample, resolved_settings.max_srt_bytes),)
                    if style_sample is not None
                    else ()
                ),
            ]
            _validate_batch_upload_sizes(
                sized_uploads,
                aggregate_limit=resolved_settings.max_batch_upload_bytes,
            )
            resolved_style, style_sample_bytes = await _resolve_batch_style(
                normalized_mode,
                style=style,
                fps=fps,
                style_sample=style_sample,
                settings=resolved_settings,
            )
            subtitle_payloads: list[bytes | None] = []
            for _, _, _, subtitle in plan:
                if subtitle is None:
                    subtitle_payloads.append(None)
                    continue
                validated_subtitle = await read_validated_srt_upload(
                    subtitle,
                    max_bytes=resolved_settings.max_srt_bytes,
                    max_line_bytes=resolved_settings.max_srt_line_bytes,
                    parse_limits=resolved_settings.srt_parse_limits,
                    label="SRT",
                )
                subtitle_payloads.append(validated_subtitle.data)

            service.store.delete_expired()
            batch_id = uuid.uuid4().hex
            jobs: list[JobRecord] = []
            tokens: list[str] = []
            directories: list[Path] = []
            succeeded = False
            total_saved = (
                (len(style_sample_bytes) if style_sample_bytes is not None else 0)
                + sum(len(payload) for payload in subtitle_payloads if payload is not None)
            )
            try:
                for position, ((source_name, audio, audio_extension, subtitle), subtitle_bytes) in enumerate(
                    zip(plan, subtitle_payloads, strict=True)
                ):
                    job_id = uuid.uuid4().hex
                    token = _unique_job_token(tokens)
                    directory = (resolved_settings.data_dir / f"job-{job_id}").resolve()
                    directory.mkdir(parents=False, exist_ok=False)
                    directories.append(directory)
                    audio_path = directory / f"audio{audio_extension}"
                    subtitle_path = directory / "original.srt" if subtitle is not None else None
                    total_saved += await _save_upload(audio, audio_path, resolved_settings.max_upload_bytes)
                    if subtitle_bytes is not None and subtitle_path is not None:
                        subtitle_path.write_bytes(subtitle_bytes)
                    if total_saved > resolved_settings.max_batch_upload_bytes:
                        raise HTTPException(status_code=413, detail="Batch uploads are too large.")
                    await reserve_processing_storage(
                        directory=directory,
                        audio_path=audio_path,
                        settings=resolved_settings,
                        store=service.store,
                        normalization_required=(
                            normalization_required
                            or selected_transcription_provider == GEMINI_TRANSCRIBE_MODEL
                        ),
                        audio_duration_probe=audio_duration_probe,
                        max_audio_duration_seconds=(
                            GEMINI_TRANSCRIBE_MAX_AUDIO_SECONDS
                            if selected_transcription_provider == GEMINI_TRANSCRIBE_MODEL
                            else None
                        ),
                    )
                    jobs.append(
                        new_job_record(
                            job_id=job_id,
                            token_hash=hash_job_token(token),
                            mode=normalized_mode,
                            directory=directory,
                            audio_path=audio_path,
                            srt_path=subtitle_path,
                            fps=fps,
                            language=language.lower(),
                            style=resolved_sync_style if normalized_mode == "sync" else resolved_style,
                            retention_hours=resolved_settings.retention_hours,
                            source_name=source_name,
                            batch_id=batch_id,
                            batch_position=position,
                            transcription_provider=selected_transcription_provider,
                        )
                    )
                    tokens.append(token)

                try:
                    service.store.create_many(
                        jobs,
                        max_outstanding=resolved_settings.max_outstanding_child_jobs,
                    )
                except OutstandingJobLimitError as exc:
                    raise HTTPException(
                        status_code=429,
                        detail="Too many files are already queued. Wait for current jobs to finish.",
                    ) from exc
                service.submit_batch(jobs)
                public_jobs: list[dict[str, object]] = []
                for job, token in zip(jobs, tokens, strict=True):
                    latest = service.store.get(job.id)
                    if latest is None:
                        raise HTTPException(status_code=500, detail="Could not create batch job.")
                    public_jobs.append(_public_job(latest, token=token))
                response = {"id": batch_id, "jobs": public_jobs}
                succeeded = True
                return response
            finally:
                if not succeeded:
                    _rollback_intake([job.id for job in jobs], directories)
        finally:
            await form.close()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, authorization: Annotated[str | None, Header()] = None) -> dict[str, object]:
        job = _authorized_job(service, job_id, authorization)
        return _public_job(job)

    @app.get("/api/jobs/{job_id}/downloads/{kind}")
    def download(job_id: str, kind: str, authorization: Annotated[str | None, Header()] = None):
        job = _authorized_job(service, job_id, authorization)
        path, filename, media_type = _download_artifact(job, kind)
        if path is None or not path.exists() or not _inside(path, job.directory):
            raise HTTPException(status_code=404, detail="Download not found.")
        return FileResponse(path, filename=filename, media_type=media_type)

    @app.post("/api/batches/{batch_id}/downloads/srt")
    async def download_batch_srts(batch_id: str, request: Request):
        credentials = await _parse_batch_download_credentials(request)
        jobs = tuple(service.store.by_batch_id(batch_id))
        _authorize_batch_download(jobs, credentials)
        if any(job.status in {"queued", "processing"} for job in jobs):
            raise HTTPException(status_code=409, detail="Batch processing is not finished.")

        completed_jobs = tuple(job for job in jobs if job.status == "complete")
        if not completed_jobs:
            raise HTTPException(status_code=409, detail="No completed SRTs are available.")
        sources = _batch_srt_sources(
            completed_jobs,
            max_source_bytes=resolved_settings.max_batch_upload_bytes,
        )
        archive = SpooledTemporaryFile(
            max_size=min(BATCH_ARCHIVE_SPOOL_BYTES, resolved_settings.max_batch_upload_bytes),
            mode="w+b",
        )
        try:
            _write_batch_srt_archive(
                archive,
                sources,
                max_source_bytes=resolved_settings.max_batch_upload_bytes,
            )
            archive.seek(0)
        except Exception:
            archive.close()
            raise

        filename = f"dubsync-batch-{batch_id[:8]}-synced-srts.zip"
        return StreamingResponse(
            _stream_and_close(archive),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    _mount_frontend(app, resolved_settings.static_dir)
    return app


def _mount_frontend(app: FastAPI, static_dir: Path) -> None:
    index = static_dir / "index.html"
    assets = static_dir / "assets"
    brand = static_dir / "brand"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    if brand.exists():
        app.mount("/brand", StaticFiles(directory=brand), name="brand")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found.")
        if full_path.endswith("/") and full_path[:-1] in FRONTEND_ROUTE_METADATA:
            return RedirectResponse(url=f"/{full_path[:-1]}", status_code=308)
        if full_path == "" and index.exists():
            return FileResponse(index, media_type="text/html")
        if full_path in FRONTEND_ROUTE_METADATA and index.exists():
            return HTMLResponse(
                _frontend_document(index, full_path),
                headers={"X-Robots-Tag": "noindex, follow"},
            )
        requested_file = static_dir / full_path
        if full_path in PUBLIC_ROOT_FILES and requested_file.is_file() and _inside(requested_file, static_dir):
            return FileResponse(requested_file)
        if not index.exists():
            return JSONResponse({"service": "dubsync", "status": "frontend_not_built"}, status_code=503)
        raise HTTPException(status_code=404, detail="Page not found.")


def _frontend_document(index: Path, route: str) -> str:
    title, description = FRONTEND_ROUTE_METADATA[route]
    canonical = f"{SITE_ORIGIN}/{route}"
    document = index.read_text(encoding="utf-8")
    replacements = (
        (r'(<meta name="description" content=")[^"]*(" />)', description),
        (r'(<meta name="robots" content=")[^"]*(" />)', "noindex, follow"),
        (r'(<link rel="canonical" href=")[^"]*(" />)', canonical),
        (r'(<meta property="og:title" content=")[^"]*(" />)', title),
        (r'(<meta property="og:description" content=")[^"]*(" />)', description),
        (r'(<meta property="og:url" content=")[^"]*(" />)', canonical),
        (r'(<meta name="twitter:title" content=")[^"]*(" />)', title),
        (r'(<meta name="twitter:description" content=")[^"]*(" />)', description),
    )
    for pattern, value in replacements:
        document = re.sub(
            pattern,
            lambda match, content=escape(value, quote=True): f"{match.group(1)}{content}{match.group(2)}",
            document,
            count=1,
        )
    document = re.sub(r"<title>.*?</title>", f"<title>{escape(title)}</title>", document, count=1, flags=re.DOTALL)
    return re.sub(
        r'\s*<script type="application/ld\+json" data-home-schema>.*?</script>',
        "",
        document,
        count=1,
        flags=re.DOTALL,
    )


def _authorized_job(service: JobService, job_id: str, authorization: str | None) -> JobRecord:
    job = service.store.get(job_id)
    token = _bearer_token(authorization)
    expired = (
        job is not None
        and job.status in {"complete", "failed"}
        and job.expires_at.timestamp() <= time.time()
    )
    if job is None or expired or not valid_job_token(token, job.token_hash):
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


async def _parse_batch_download_credentials(request: Request) -> tuple[tuple[str, str], ...]:
    content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json.")

    body = b""
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BATCH_DOWNLOAD_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Batch download request body is too large.")
        body = body + chunk

    try:
        payload = json.loads(body.decode("utf-8"), object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise HTTPException(status_code=422, detail="Invalid batch download request.") from exc
    if not isinstance(payload, dict) or set(payload) != {"jobs"}:
        raise HTTPException(status_code=422, detail="Invalid batch download request.")

    requested_jobs = payload["jobs"]
    if (
        not isinstance(requested_jobs, list)
        or not requested_jobs
        or len(requested_jobs) > MAX_BATCH_ITEMS
    ):
        raise HTTPException(status_code=422, detail="Invalid batch download request.")

    credentials: tuple[tuple[str, str], ...] = ()
    for requested_job in requested_jobs:
        if not isinstance(requested_job, dict) or set(requested_job) != {"id", "token"}:
            raise HTTPException(status_code=422, detail="Invalid batch download request.")
        job_id = requested_job["id"]
        token = requested_job["token"]
        if (
            not isinstance(job_id, str)
            or not job_id
            or len(job_id) > 256
            or not isinstance(token, str)
            or not token
            or len(token) > 1024
        ):
            raise HTTPException(status_code=422, detail="Invalid batch download request.")
        credentials = (*credentials, (job_id, token))
    return credentials


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("Duplicate JSON field")
        payload = {**payload, key: value}
    return payload


def _authorize_batch_download(
    jobs: tuple[JobRecord, ...],
    credentials: tuple[tuple[str, str], ...],
) -> None:
    if not jobs:
        raise HTTPException(status_code=404, detail="Batch not found.")

    provided_ids = tuple(job_id for job_id, _token in credentials)
    expected_ids = tuple(job.id for job in jobs)
    ids_match = (
        len(provided_ids) == len(expected_ids)
        and len(frozenset(provided_ids)) == len(provided_ids)
        and frozenset(provided_ids) == frozenset(expected_ids)
    )
    tokens_match = True
    for job in jobs:
        matching_tokens = tuple(token for job_id, token in credentials if job_id == job.id)
        supplied_token = matching_tokens[0] if len(matching_tokens) == 1 else ""
        tokens_match = valid_job_token(supplied_token, job.token_hash) and tokens_match

    now = time.time()
    has_expired_child = any(
        job.status in {"complete", "failed"} and job.expires_at.timestamp() <= now
        for job in jobs
    )
    if not ids_match or not tokens_match or has_expired_child:
        raise HTTPException(status_code=404, detail="Batch not found.")


def _batch_srt_sources(
    jobs: tuple[JobRecord, ...],
    *,
    max_source_bytes: int,
) -> tuple[tuple[Path, str], ...]:
    sources: tuple[tuple[Path, str], ...] = ()
    source_bytes = 0
    filenames: tuple[str, ...] = ()
    for job in jobs:
        path, filename, _media_type = _download_artifact(job, "srt")
        if (
            path is None
            or not _safe_archive_filename(filename)
            or filename in filenames
        ):
            raise HTTPException(status_code=404, detail="Download not found.")
        try:
            resolved_path = path.resolve(strict=True)
            size = resolved_path.stat().st_size
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Download not found.") from exc
        if not resolved_path.is_file() or not _inside(resolved_path, job.directory):
            raise HTTPException(status_code=404, detail="Download not found.")
        source_bytes += size
        if source_bytes > max_source_bytes:
            raise HTTPException(status_code=413, detail="Batch SRT archive is too large.")
        sources = (*sources, (resolved_path, filename))
        filenames = (*filenames, filename)
    return sources


def _safe_archive_filename(filename: str) -> bool:
    return bool(filename) and filename not in {".", ".."} and "/" not in filename and "\\" not in filename


def _write_batch_srt_archive(
    archive: BinaryIO,
    sources: tuple[tuple[Path, str], ...],
    *,
    max_source_bytes: int,
) -> None:
    source_bytes = 0
    try:
        with ZipFile(archive, mode="w", compression=ZIP_DEFLATED) as zip_file:
            for path, filename in sources:
                with path.open("rb") as source, zip_file.open(filename, mode="w") as target:
                    while chunk := source.read(ARCHIVE_COPY_CHUNK_BYTES):
                        source_bytes += len(chunk)
                        if source_bytes > max_source_bytes:
                            raise HTTPException(status_code=413, detail="Batch SRT archive is too large.")
                        target.write(chunk)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Download not found.") from exc


def _stream_and_close(archive: BinaryIO) -> Iterator[bytes]:
    try:
        while chunk := archive.read(ARCHIVE_COPY_CHUNK_BYTES):
            yield chunk
    finally:
        archive.close()


async def _parse_single_form(request: Request) -> FormData:
    try:
        return await request.form(
            max_files=MAX_SINGLE_PARSER_FILES,
            max_fields=MAX_SINGLE_PARSER_FIELDS,
            max_part_size=MAX_BATCH_FIELD_BYTES,
        )
    except StarletteHTTPException as exc:
        detail = str(exc.detail)
        if detail.startswith("Too many files.") or detail.startswith("Too many fields."):
            raise HTTPException(
                status_code=422,
                detail="Single jobs accept one audio file, one original SRT, and one style example.",
            ) from exc
        raise


def _validate_single_form_shape(form: FormData) -> None:
    allowed_fields = {
        "mode",
        "fps",
        "language",
        "style",
        "sync_max_lines_per_cue",
        "transcription_provider",
    }
    allowed_files = {"audio", "subtitle", "style_sample"}
    for name, value in form.multi_items():
        allowed = allowed_files if isinstance(value, StarletteUploadFile) else allowed_fields
        if name not in allowed:
            raise HTTPException(status_code=422, detail="Unexpected job form field.")


async def _parse_batch_form(request: Request) -> FormData:
    try:
        return await request.form(
            max_files=MAX_BATCH_PARSER_FILES,
            max_fields=MAX_BATCH_PARSER_FIELDS,
            max_part_size=MAX_BATCH_FIELD_BYTES,
        )
    except StarletteHTTPException as exc:
        if str(exc.detail).startswith("Too many files."):
            raise HTTPException(status_code=422, detail="Select no more than 10 file pairs.") from exc
        raise


def _validate_batch_form_shape(form: FormData) -> None:
    allowed_fields = {
        "mode",
        "fps",
        "language",
        "style",
        "sync_max_lines_per_cue",
        "transcription_provider",
    }
    allowed_files = {"audio", "subtitle", "style_sample"}
    for name, value in form.multi_items():
        allowed = allowed_files if isinstance(value, StarletteUploadFile) else allowed_fields
        if name not in allowed:
            raise HTTPException(status_code=422, detail="Unexpected batch form field.")


def _batch_text_field(form: FormData, name: str, *, default: str | None = None) -> str:
    values = form.getlist(name)
    if not values:
        if default is None:
            raise HTTPException(status_code=422, detail=f"Missing {name} field.")
        return default
    if len(values) != 1 or not isinstance(values[0], str):
        raise HTTPException(status_code=422, detail=f"Invalid {name} field.")
    value = values[0]
    if len(value.encode("utf-8")) > MAX_BATCH_FIELD_BYTES:
        raise HTTPException(status_code=422, detail=f"The {name} field is too large.")
    return value


def _batch_fps_field(form: FormData, *, mode: JobMode) -> float | None:
    value = _batch_text_field(form, "fps", default="").strip()
    if not value:
        return None if mode == "sync" else 30.0
    try:
        return float(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid fps field.") from exc


def _batch_sync_max_lines_field(form: FormData) -> int | None:
    value = _batch_text_field(form, "sync_max_lines_per_cue", default="").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid sync maximum lines field.") from exc
    if str(parsed) != value or not SYNC_MAX_LINES_MIN <= parsed <= SYNC_MAX_LINES_MAX:
        raise HTTPException(status_code=422, detail="Sync maximum lines per cue must be 1 or 2.")
    return parsed


def _sync_style_for_mode(mode: JobMode, max_lines_per_cue: int | None) -> str:
    if mode == "generate":
        if max_lines_per_cue is not None:
            raise HTTPException(status_code=422, detail="Sync maximum lines only applies to sync mode.")
        return "standard"
    return sync_style_from_max_lines(max_lines_per_cue)


def _batch_file_field(form: FormData, name: str) -> list[StarletteUploadFile]:
    values = form.getlist(name)
    if any(not isinstance(value, StarletteUploadFile) for value in values):
        raise HTTPException(status_code=422, detail=f"Invalid {name} upload field.")
    return list(values)  # type: ignore[return-value]


def _validate_batch_upload_sizes(
    uploads: list[tuple[StarletteUploadFile, int]],
    *,
    aggregate_limit: int,
) -> None:
    aggregate_size = 0
    for upload, per_file_limit in uploads:
        size = upload.size
        if size is None:
            continue
        if size <= 0:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")
        if size > per_file_limit:
            raise HTTPException(status_code=413, detail="Uploaded file is too large.")
        aggregate_size += size
        if aggregate_size > aggregate_limit:
            raise HTTPException(status_code=413, detail="Batch uploads are too large.")


async def _resolve_batch_style(
    mode: JobMode,
    *,
    style: str,
    fps: float | None,
    style_sample: StarletteUploadFile | None,
    settings: WebSettings,
) -> tuple[str, bytes | None]:
    if mode == "sync":
        if style_sample is not None:
            raise HTTPException(status_code=422, detail="Sync batches do not accept a style example.")
        return "source", None

    generation_fps = _require_generation_fps(fps)

    try:
        style_request = parse_generation_style_request(style)
    except GenerationStyleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    sample_cues = None
    style_sample_bytes: bytes | None = None
    if style_request.source == "sample":
        if style_sample is None:
            raise HTTPException(status_code=422, detail="An SRT style example is required.")
        _validate_subtitle(style_sample)
        validated_style = await read_validated_srt_upload(
            style_sample,
            max_bytes=settings.max_srt_bytes,
            max_line_bytes=settings.max_srt_line_bytes,
            parse_limits=settings.srt_parse_limits,
            label="SRT style example",
        )
        style_sample_bytes = validated_style.data
        sample_cues = list(validated_style.cues)
    elif style_sample is not None:
        raise HTTPException(
            status_code=422,
            detail="A style example is only allowed with the sample style option.",
        )

    try:
        resolved_style = resolve_generation_style(
            style_request,
            fps=generation_fps,
            sample_cues=sample_cues,
        ).model_dump_json()
    except GenerationStyleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return resolved_style, style_sample_bytes


def _unique_job_token(existing: list[str]) -> str:
    existing_tokens = set(existing)
    while token := secrets.token_urlsafe(32):
        if token not in existing_tokens:
            return token
    raise RuntimeError("Could not generate a job token")  # pragma: no cover


def _validate_mode(mode: str) -> JobMode:
    normalized = mode.strip().lower()
    if normalized not in {"sync", "generate"}:
        raise HTTPException(status_code=422, detail="Mode must be sync or generate.")
    return normalized  # type: ignore[return-value]


def _require_generation_fps(fps: float | None) -> float:
    if fps is None:
        raise HTTPException(status_code=422, detail="A frame rate is required for generate mode.")
    return fps


def _validate_options(*, mode: JobMode, fps: float | None, language: str) -> None:
    if fps is None and mode != "sync":
        raise HTTPException(status_code=422, detail="A frame rate is required for generate mode.")
    if fps is not None and fps not in FPS_VALUES:
        raise HTTPException(status_code=422, detail="Unsupported frame rate.")
    if not LANGUAGE_RE.fullmatch(language.strip()):
        raise HTTPException(status_code=422, detail="Invalid language code.")


def _validate_transcription_provider(value: str, *, settings: WebSettings) -> str:
    normalized = value.strip().lower() or DEFAULT_TRANSCRIPTION_PROVIDER
    if normalized not in TRANSCRIPTION_PROVIDER_VALUES:
        raise HTTPException(status_code=422, detail="Invalid transcription provider.")
    if normalized == GEMINI_TRANSCRIBE_MODEL:
        raise HTTPException(
            status_code=422,
            detail=GEMINI_TRANSCRIBE_DISABLED_MESSAGE,
        )
    return normalized


def _gemini_transcribe_web_available(settings: WebSettings) -> bool:
    del settings
    return False


async def _save_upload(upload: UploadFile | StarletteUploadFile, destination: Path, limit: int) -> int:
    size = 0
    try:
        with destination.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large.")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    return size


def _public_job(job: JobRecord, *, token: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": job.id,
        "mode": job.mode,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "expires_at": job.expires_at.isoformat(),
        "error": job.error,
        "source_name": job.source_name,
        "batch_id": job.batch_id,
        "batch_position": job.batch_position,
        "transcription_provider": job.transcription_provider,
        "result": None,
        "downloads": [],
    }
    if token is not None:
        payload["token"] = token
    if job.status == "complete":
        payload["result"] = {
            "cue_count": job.cue_count,
            "cost_usd": job.cost_usd,
            **_qc_result_metadata(job.qc_json),
        }
        downloads = ["srt", "qc-json", "qc-html"]
        if job.changes_srt is not None:
            downloads.append("changes")
        payload["downloads"] = downloads
    return payload


def _qc_result_metadata(qc_json: Path | None) -> dict[str, object]:
    if qc_json is None:
        return {}
    try:
        if qc_json.stat().st_size > MAX_QC_RESULT_METADATA_BYTES:
            return {}
        payload = json.loads(qc_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
        return {}

    summary = payload["summary"]
    raw_fps = summary.get("fps")
    raw_source = summary.get("fps_source")
    raw_confidence = summary.get("fps_detection_confident")
    if (
        isinstance(raw_fps, bool)
        or not isinstance(raw_fps, int | float)
        or not 0 < float(raw_fps) <= 240
        or not isinstance(raw_source, str)
        or raw_source not in FPS_RESULT_SOURCES
        or not isinstance(raw_confidence, bool)
    ):
        return {}
    return {
        "fps": float(raw_fps),
        "fps_source": raw_source,
        "fps_detection_confident": raw_confidence,
    }


def _download_artifact(job: JobRecord, kind: str) -> tuple[Path | None, str, str]:
    output_filename = (
        f"{job.source_name}-dubsync-synced.srt"
        if job.source_name is not None
        else ("dubsync.generated.srt" if job.mode == "generate" else "dubsync.synced.srt")
    )
    artifacts = {
        "srt": (job.output_srt, output_filename, "application/x-subrip"),
        "qc-json": (job.qc_json, "dubsync.qc.json", "application/json"),
        "qc-html": (job.qc_html, "dubsync.qc.html", "text/html"),
        "changes": (job.changes_srt, "dubsync.changes.srt", "application/x-subrip"),
    }
    if kind not in artifacts:
        raise HTTPException(status_code=404, detail="Download not found.")
    return artifacts[kind]


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def run() -> None:
    import uvicorn

    uvicorn.run(
        "dubsync.web.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
