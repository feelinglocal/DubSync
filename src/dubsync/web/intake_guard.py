from __future__ import annotations

import os
import re
import secrets
import shutil
import threading
from collections.abc import Callable
from ipaddress import ip_address

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.formparsers import MultiPartException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .security import SlidingWindowRateLimiter
from .settings import WebSettings

MAX_MULTIPART_OVERHEAD_BYTES = 256 * 1024
JOB_INTAKE_PATHS = frozenset({"/api/jobs", "/api/batches"})
BATCH_BODY_LIMIT_DETAIL = "Batch request body is too large."
SINGLE_BODY_LIMIT_DETAIL = "Request body is too large."
STORAGE_CAPACITY_DETAIL = "Storage capacity is temporarily unavailable. Wait for existing jobs to expire."


def job_intake_preflight(
    request: Request,
    *,
    settings: WebSettings,
    limiter: SlidingWindowRateLimiter,
) -> JSONResponse | None:
    try:
        _validate_job_access(settings, request.headers.get("x-dubsync-access-code", ""))
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    if not limiter.allow(_client_key(request)):
        return JSONResponse(
            {"detail": "Too many jobs. Try again later."},
            status_code=429,
        )

    content_lengths = request.headers.getlist("content-length")
    if content_lengths:
        if len(content_lengths) != 1 or re.fullmatch(r"[0-9]+", content_lengths[0]) is None:
            return JSONResponse(
                {"detail": "Invalid Content-Length header."},
                status_code=400,
            )
        body_limit, detail = job_intake_body_limit(request.url.path, settings)
        declared_length = content_lengths[0].lstrip("0") or "0"
        body_limit_text = str(body_limit)
        if len(declared_length) > len(body_limit_text) or (
            len(declared_length) == len(body_limit_text)
            and declared_length > body_limit_text
        ):
            return JSONResponse({"detail": detail}, status_code=413)
    return None


def job_intake_body_limit(path: str, settings: WebSettings) -> tuple[int, str]:
    if path == "/api/batches":
        return (
            settings.max_batch_upload_bytes + MAX_MULTIPART_OVERHEAD_BYTES,
            BATCH_BODY_LIMIT_DETAIL,
        )
    return (
        settings.max_upload_bytes + (2 * settings.max_srt_bytes) + MAX_MULTIPART_OVERHEAD_BYTES,
        SINGLE_BODY_LIMIT_DETAIL,
    )


def _limited_receive(receive: Receive, *, body_limit: int, detail: str) -> Receive:
    received = 0

    async def limited_receive() -> Message:
        nonlocal received
        message = await receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > body_limit:
                raise MultiPartException(detail)
        return message

    return limited_receive


class SecurityAndIntakeMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: WebSettings,
        limiter: SlidingWindowRateLimiter,
        storage_usage_bytes: Callable[[], int],
    ) -> None:
        self.app = app
        self.settings = settings
        self.limiter = limiter
        self.storage_usage_bytes = storage_usage_bytes
        self.intake_lock = threading.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
                headers["Content-Security-Policy"] = (
                    "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; "
                    "script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
                    "connect-src 'self'; media-src 'self' blob:; worker-src 'self' blob:; form-action 'self'"
                )
                if path.startswith("/api/"):
                    headers["Cache-Control"] = "no-store"
            await send(message)

        request = Request(scope, receive=receive)
        guarded_receive = receive
        if request.method == "POST" and path in JOB_INTAKE_PATHS:
            try:
                _validate_job_access(self.settings, request.headers.get("x-dubsync-access-code", ""))
            except HTTPException as exc:
                response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
                await response(scope, receive, send_with_security_headers)
                return
            if not self.intake_lock.acquire(blocking=False):
                response = JSONResponse(
                    {"detail": "Another upload is already being accepted. Try again shortly."},
                    status_code=429,
                )
                await response(scope, receive, send_with_security_headers)
                return
            try:
                response = job_intake_preflight(
                    request,
                    settings=self.settings,
                    limiter=self.limiter,
                )
                if response is not None:
                    await response(scope, receive, send_with_security_headers)
                    return
                reservation = _request_body_reservation(request, path=path, settings=self.settings)
                try:
                    retained_bytes = self.storage_usage_bytes()
                except OSError:
                    response = JSONResponse(
                        {"detail": "Storage capacity could not be verified."},
                        status_code=503,
                    )
                    await response(scope, receive, send_with_security_headers)
                    return
                if retained_bytes + reservation > self.settings.max_retained_storage_bytes:
                    response = JSONResponse(
                        {"detail": STORAGE_CAPACITY_DETAIL},
                        status_code=507,
                    )
                    await response(scope, receive, send_with_security_headers)
                    return
                try:
                    free_bytes = shutil.disk_usage(self.settings.data_dir).free
                except OSError:
                    response = JSONResponse(
                        {"detail": "Storage capacity could not be verified."},
                        status_code=503,
                    )
                    await response(scope, receive, send_with_security_headers)
                    return
                if free_bytes < reservation + self.settings.min_free_storage_bytes:
                    response = JSONResponse(
                        {"detail": STORAGE_CAPACITY_DETAIL},
                        status_code=507,
                    )
                    await response(scope, receive, send_with_security_headers)
                    return
                body_limit, body_limit_detail = job_intake_body_limit(path, self.settings)
                guarded_receive = _limited_receive(
                    receive,
                    body_limit=body_limit,
                    detail=body_limit_detail,
                )
                await self.app(scope, guarded_receive, send_with_security_headers)
            finally:
                self.intake_lock.release()
            return

        await self.app(scope, guarded_receive, send_with_security_headers)


def _validate_job_access(settings: WebSettings, access_code: str) -> None:
    if settings.require_job_access_code and not settings.job_access_code:
        raise HTTPException(status_code=503, detail="Job access is not configured.")
    if settings.job_access_code and not secrets.compare_digest(access_code.strip(), settings.job_access_code):
        raise HTTPException(status_code=403, detail="A valid job access code is required.")


def _client_key(request: Request) -> str:
    if os.getenv("RENDER", "").strip().lower() == "true":
        forwarded = [
            part.strip()
            for part in request.headers.get("x-forwarded-for", "").split(",")
            if part.strip()
        ]
        if forwarded:
            try:
                return str(ip_address(forwarded[0]))
            except ValueError:
                pass
    return request.client.host if request.client else "unknown"


def _request_body_reservation(request: Request, *, path: str, settings: WebSettings) -> int:
    content_lengths = request.headers.getlist("content-length")
    if not content_lengths:
        return job_intake_body_limit(path, settings)[0]
    normalized = content_lengths[0].lstrip("0") or "0"
    return int(normalized)
