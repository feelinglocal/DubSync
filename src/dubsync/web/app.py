from __future__ import annotations

import os
import re
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..srt_io import SRTParseError, parse_srt_text
from .generation_styles import (
    GenerationStyleError,
    parse_generation_style_request,
    public_generation_styles,
    resolve_generation_style,
)
from .jobs import JobMode, JobRecord, JobService, Processor, default_processor, new_job_record
from .security import SlidingWindowRateLimiter, hash_job_token, valid_job_token
from .settings import WebSettings

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
AUDIO_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/wave",
    "audio/x-m4a",
    "audio/x-wav",
    "application/octet-stream",
}
FPS_VALUES = {23.976, 24.0, 25.0, 29.97, 30.0}
LANGUAGE_RE = re.compile(r"^(auto|[A-Za-z]{2,8}(?:-[A-Za-z]{2,8})?)$")


def create_app(
    *,
    settings: WebSettings | None = None,
    processor: Processor = default_processor,
) -> FastAPI:
    resolved_settings = settings or WebSettings.from_env()
    resolved_settings.ensure_directories()
    service = JobService(resolved_settings, processor)
    limiter = SlidingWindowRateLimiter(resolved_settings.max_jobs_per_hour)

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

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; "
            "script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
            "connect-src 'self'; media-src 'self' blob:; worker-src 'self' blob:; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

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
            "audio_extensions": sorted(AUDIO_EXTENSIONS),
            "fps_values": sorted(FPS_VALUES),
            "pricing": {
                "generate": {"usd_per_minute": 0.12, "minimum_usd": 3.0},
                "sync": {"usd_per_minute": 0.18, "minimum_usd": 5.0},
                "precision": {"usd_per_minute": 0.25, "minimum_usd": 10.0},
            },
            "billing_enabled": False,
            "access_code_required": access_code_required,
            "jobs_available": jobs_available,
            "generation_styles": public_generation_styles(),
        }

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        request: Request,
        mode: Annotated[str, Form()],
        audio: Annotated[UploadFile, File()],
        subtitle: Annotated[UploadFile | None, File()] = None,
        style_sample: Annotated[UploadFile | None, File()] = None,
        fps: Annotated[float, Form()] = 30.0,
        language: Annotated[str, Form()] = "auto",
        style: Annotated[str, Form()] = "standard",
        access_code: Annotated[str, Form()] = "",
    ) -> dict[str, object]:
        client_key = _client_key(request)
        if not limiter.allow(client_key):
            raise HTTPException(status_code=429, detail="Too many jobs. Try again later.")
        if resolved_settings.require_job_access_code and not resolved_settings.job_access_code:
            raise HTTPException(status_code=503, detail="Job access is not configured.")
        if resolved_settings.job_access_code and not secrets.compare_digest(
            access_code.strip(), resolved_settings.job_access_code
        ):
            raise HTTPException(status_code=403, detail="A valid job access code is required.")
        normalized_mode = _validate_mode(mode)
        if normalized_mode == "sync" and subtitle is None:
            raise HTTPException(status_code=422, detail="An original SRT is required for sync mode.")
        _validate_options(fps=fps, language=language)
        audio_extension = _validate_audio(audio)
        if subtitle is not None:
            _validate_subtitle(subtitle)

        resolved_style = "source"
        style_sample_bytes: bytes | None = None
        if normalized_mode == "generate":
            try:
                style_request = parse_generation_style_request(style)
            except GenerationStyleError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            sample_cues = None
            if style_request.source == "sample":
                if style_sample is None:
                    raise HTTPException(status_code=422, detail="An SRT style example is required.")
                _validate_subtitle(style_sample)
                style_sample_bytes = await _read_upload(style_sample, resolved_settings.max_srt_bytes)
                try:
                    sample_text = style_sample_bytes.decode("utf-8-sig")
                    sample_cues = parse_srt_text(sample_text)
                    if not sample_cues:
                        raise SRTParseError("no subtitle cues were found")
                except (UnicodeDecodeError, SRTParseError, ValueError) as exc:
                    detail = str(exc).splitlines()[0] or "invalid SRT"
                    raise HTTPException(status_code=422, detail=f"Could not read the SRT style example: {detail}") from exc
            elif style_sample is not None:
                await style_sample.close()
            try:
                resolved_style = resolve_generation_style(
                    style_request,
                    fps=fps,
                    sample_cues=sample_cues,
                ).model_dump_json()
            except GenerationStyleError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        elif style_sample is not None:
            await style_sample.close()

        service.store.delete_expired()
        job_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        directory = (resolved_settings.data_dir / f"job-{job_id}").resolve()
        directory.mkdir(parents=False, exist_ok=False)
        audio_path = directory / f"audio{audio_extension}"
        subtitle_path = directory / "original.srt" if subtitle is not None else None
        try:
            await _save_upload(audio, audio_path, resolved_settings.max_upload_bytes)
            if subtitle is not None and subtitle_path is not None:
                await _save_upload(subtitle, subtitle_path, resolved_settings.max_srt_bytes)
            if style_sample_bytes is not None:
                (directory / "style-example.srt").write_bytes(style_sample_bytes)
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
            )
            service.store.create(job)
            service.submit(job)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        latest = service.store.get(job_id)
        if latest is None:
            raise HTTPException(status_code=500, detail="Could not create job.")
        return _public_job(latest, token=token)

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

    _mount_frontend(app, resolved_settings.static_dir)
    return app


def _mount_frontend(app: FastAPI, static_dir: Path) -> None:
    index = static_dir / "index.html"
    assets = static_dir / "assets"
    frontend_routes = {"", "terms", "privacy", "payments"}
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found.")
        if full_path in frontend_routes and index.exists():
            return FileResponse(index, media_type="text/html")
        requested_file = static_dir / full_path
        if requested_file.is_file() and _inside(requested_file, static_dir):
            return FileResponse(requested_file)
        if not index.exists():
            return JSONResponse({"service": "dubsync", "status": "frontend_not_built"}, status_code=503)
        raise HTTPException(status_code=404, detail="Page not found.")


def _authorized_job(service: JobService, job_id: str, authorization: str | None) -> JobRecord:
    job = service.store.get(job_id)
    token = _bearer_token(authorization)
    if job is None or job.expires_at.timestamp() <= time.time() or not valid_job_token(token, job.token_hash):
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _validate_mode(mode: str) -> JobMode:
    normalized = mode.strip().lower()
    if normalized not in {"sync", "generate"}:
        raise HTTPException(status_code=422, detail="Mode must be sync or generate.")
    return normalized  # type: ignore[return-value]


def _validate_options(*, fps: float, language: str) -> None:
    if fps not in FPS_VALUES:
        raise HTTPException(status_code=422, detail="Unsupported frame rate.")
    if not LANGUAGE_RE.fullmatch(language.strip()):
        raise HTTPException(status_code=422, detail="Invalid language code.")


def _validate_audio(upload: UploadFile) -> str:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in AUDIO_EXTENSIONS or upload.content_type not in AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio file.")
    return extension


def _validate_subtitle(upload: UploadFile) -> None:
    if Path(upload.filename or "").suffix.lower() != ".srt":
        raise HTTPException(status_code=415, detail="Subtitle must be an SRT file.")
    if upload.content_type not in {"application/octet-stream", "application/x-subrip", "text/plain"}:
        raise HTTPException(status_code=415, detail="Unsupported subtitle file.")


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> None:
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


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    data = bytearray()
    try:
        while chunk := await upload.read(1024 * 1024):
            data.extend(chunk)
            if len(data) > limit:
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
    finally:
        await upload.close()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    return bytes(data)


def _public_job(job: JobRecord, *, token: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": job.id,
        "mode": job.mode,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "expires_at": job.expires_at.isoformat(),
        "error": job.error,
        "result": None,
        "downloads": [],
    }
    if token is not None:
        payload["token"] = token
    if job.status == "complete":
        payload["result"] = {"cue_count": job.cue_count, "cost_usd": job.cost_usd}
        downloads = ["srt", "qc-json", "qc-html"]
        if job.changes_srt is not None:
            downloads.append("changes")
        payload["downloads"] = downloads
    return payload


def _download_artifact(job: JobRecord, kind: str) -> tuple[Path | None, str, str]:
    artifacts = {
        "srt": (job.output_srt, "dubsync.generated.srt" if job.mode == "generate" else "dubsync.synced.srt", "application/x-subrip"),
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


def _client_key(request: Request) -> str:
    if os.getenv("RENDER", "").strip().lower() == "true":
        forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
        if forwarded:
            try:
                return str(ip_address(forwarded[-1]))
            except ValueError:
                pass
    return request.client.host if request.client else "unknown"


def run() -> None:
    import uvicorn

    uvicorn.run(
        "dubsync.web.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
