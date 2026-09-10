from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from dubsync.gemini_audio_context import GeminiAudioContext
from dubsync.providers import ProviderError


class FakeFiles:
    def __init__(self):
        self.uploads = []
        self.deleted = []
        self.state = "ACTIVE"

    def upload(self, *, file, config):
        self.uploads.append((file, config))
        return SimpleNamespace(name="files/owned", uri="https://example.test/owned", state=self.state)

    def get(self, *, name):
        return SimpleNamespace(name=name, uri="https://example.test/owned", state=self.state)

    def delete(self, *, name):
        self.deleted.append(name)


class FakeCaches:
    def __init__(self):
        self.created = []
        self.updated = []
        self.deleted = []
        self.error = None

    def create(self, *, model, config):
        self.created.append((model, config))
        if self.error:
            raise self.error
        return SimpleNamespace(name="cachedContents/owned", usage_metadata=SimpleNamespace(total_token_count=10000))

    def update(self, *, name, config):
        self.updated.append((name, config))

    def delete(self, *, name):
        self.deleted.append(name)


@pytest.fixture
def audio_context(monkeypatch, tmp_path):
    import dubsync.gemini_audio_context as module

    clients = []
    clock = [0.0]

    def factory(**kwargs):
        client = SimpleNamespace(files=FakeFiles(), caches=FakeCaches(), closed=False, options=kwargs)
        client.close = lambda: setattr(client, "closed", True)
        clients.append(client)
        return client

    monkeypatch.setattr(module, "_new_client", factory)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module, "_utc_now", lambda: datetime(2026, 9, 10, tzinfo=timezone.utc) + timedelta(seconds=clock[0]))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    path = tmp_path / "episode.mp3"
    path.write_bytes(b"original mp3 bytes")

    def make(duration=300, **config):
        return GeminiAudioContext(api_key="private-key", model="gemini-3.8-flash", path=path,
                                  duration_seconds=duration, config=config)

    return make, clients, path, clock


def test_original_media_is_lazy_uploaded_once_and_cached_across_requests(audio_context, monkeypatch):
    make, clients, path, _ = audio_context
    context = make()
    assert not clients
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("full audio must be streamed"))
    first = context.for_request()
    second = context.for_request()
    assert first.cached_content == second.cached_content == "cachedContents/owned"
    assert first.file_uri is None
    assert len(clients) == 1
    client = clients[0]
    assert client.files.uploads[0][0] == path
    assert client.files.uploads[0][1]["mime_type"] == "audio/mpeg"
    assert len(client.files.uploads) == len(client.caches.created) == 1
    assert client.options["http_options"]["retry_options"]["attempts"] == 1
    assert context.report()["original_sha256"] == hashlib.sha256(b"original mp3 bytes").hexdigest()
    assert context.report()["cached_requests"] == 2
    context.close()
    context.close()
    assert client.caches.deleted == ["cachedContents/owned"]
    assert client.files.deleted == ["files/owned"]
    assert client.closed
    assert context.report()["cleanup_status"] == "complete"


def test_short_wav_keeps_original_format_and_bounds_uncached_input(audio_context):
    make, clients, path, _ = audio_context
    wav = path.with_suffix(".wav")
    path.rename(wav)
    context = GeminiAudioContext(api_key="private-key", model="gemini-3.8-flash", path=wav,
                                 duration_seconds=60, config={"max_uncached_audio_tokens": 3840})
    assert context.for_request().file_uri == "https://example.test/owned"
    assert context.for_request().file_uri == "https://example.test/owned"
    with pytest.raises(ProviderError, match="uncached input budget"):
        context.for_request()
    assert clients[0].files.uploads[0][1]["mime_type"] == "audio/wav"
    assert clients[0].caches.created == []
    assert context.report()["uncached_audio_tokens_reserved"] == 3840
    assert context.report()["degraded_requests"] == 1
    assert "uncached_budget_exhausted" in context.report()["warnings"]
    context.close()


def test_cache_failure_uses_bounded_uri_fallback_without_duplicate_cache_creation(audio_context):
    make, clients, _, _ = audio_context
    context = make(max_uncached_audio_tokens=9600)
    context._open_client()
    clients[0].caches.error = RuntimeError("unsupported model - private body")
    assert context.for_request().file_uri
    with pytest.raises(ProviderError, match="uncached input budget"):
        context.for_request()
    assert len(clients[0].caches.created) == 1
    assert "cache_unavailable" in context.report()["warnings"]
    assert "private body" not in str(context.report())
    context.close()
    assert clients[0].files.deleted == ["files/owned"]


def test_processing_timeout_deletes_owned_file_and_degrades_explicitly(audio_context):
    make, clients, _, clock = audio_context
    context = make(upload_timeout_seconds=2)
    context._open_client()
    clients[0].files.state = "PROCESSING"
    with pytest.raises(ProviderError, match="could not be prepared"):
        context.for_request()
    assert clock[0] <= 2
    assert clients[0].files.deleted == ["files/owned"]
    assert "upload_unavailable" in context.report()["warnings"]
    context.close()


def test_cache_lease_is_refreshed_but_never_beyond_job_lifetime(audio_context):
    make, clients, _, clock = audio_context
    context = make(cache_ttl_seconds=120, max_lifetime_seconds=240)
    assert context.for_request().cached_content
    clock[0] = 70
    assert context.for_request().cached_content
    assert clients[0].caches.updated == [("cachedContents/owned", {"expire_time": datetime(2026, 9, 10, tzinfo=timezone.utc) + timedelta(seconds=190)})]
    clock[0] = 241
    with pytest.raises(ProviderError, match="job lifetime"):
        context.for_request()
    assert clients[0].caches.deleted == ["cachedContents/owned"]
    assert clients[0].files.deleted == ["files/owned"]
    assert "context_lifetime_exhausted" in context.report()["warnings"]


def test_invalid_media_is_rejected_before_upload(audio_context):
    make, clients, _, _ = audio_context
    with pytest.raises(ProviderError, match="size"):
        make(max_audio_bytes=1)
    with pytest.raises(ProviderError, match="duration"):
        make(duration=float("nan"))
    assert not clients


def test_cleanup_failure_is_explicit_and_does_not_leak_exception_body(audio_context):
    make, clients, _, _ = audio_context
    context = make()
    context.for_request()

    def fail(**kwargs):
        raise RuntimeError("private server detail")

    clients[0].caches.delete = fail
    context.close()
    assert clients[0].files.deleted == ["files/owned"]
    assert clients[0].closed
    assert context.report()["cleanup_status"] == "incomplete"
    assert "cache_delete_failed" in context.report()["warnings"]
    assert "private server detail" not in str(context.report())


def test_long_audio_compresses_one_owned_copy_preserving_original(audio_context, monkeypatch):
    import dubsync.gemini_audio_context as module

    make, clients, path, _ = audio_context
    source = b"s" * (3 * 1024 * 1024)
    path.write_bytes(source)
    conversions = []

    def convert(command, **options):
        conversions.append((command, options))
        Path(command[-1]).write_bytes(b"compressed speech")

    monkeypatch.setattr(module.subprocess, "run", convert)
    monkeypatch.setattr(module, "probe_audio_duration", lambda *_: 300.02)
    context = make()
    context.for_request()
    context.for_request()
    upload = clients[0].files.uploads[0][0]
    assert upload != path
    assert upload.parent.name.startswith("dubsync-gemini-")
    assert upload.name.startswith(hashlib.sha256(source).hexdigest()[:24])
    assert len(conversions) == len(clients[0].files.uploads) == 1
    assert path.read_bytes() == source
    assert context.report()["compression"] == "mp3_mono_64kbps_24khz"
    assert context.report()["uploaded_size_bytes"] < context.report()["original_size_bytes"]
    assert context.report()["original_sha256"] != context.report()["uploaded_sha256"]
    context.close()
    assert not upload.exists()
    assert path.read_bytes() == source


def test_compact_mp3_skips_reencoding(audio_context, monkeypatch):
    import dubsync.gemini_audio_context as module

    make, clients, path, _ = audio_context
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must reuse compact MP3"))
    context = make()
    context.for_request()
    assert clients[0].files.uploads[0][0] == path
    assert context.report()["compression"] == "reused_compact_mp3"
    context.close()


def test_failed_compression_cleans_partial_copy_and_never_uploads(audio_context, monkeypatch):
    import dubsync.gemini_audio_context as module

    make, clients, path, _ = audio_context
    source = b"s" * (3 * 1024 * 1024)
    path.write_bytes(source)
    output = []

    def fail(command, **options):
        output.append(Path(command[-1]))
        output[-1].write_bytes(b"partial")
        raise module.subprocess.TimeoutExpired(command, options["timeout"])

    monkeypatch.setattr(module.subprocess, "run", fail)
    context = make()
    with pytest.raises(ProviderError, match="could not be prepared"):
        context.for_request()
    assert not clients
    assert not output[0].exists()
    assert path.read_bytes() == source
    assert context.report()["cleanup_status"] == "complete"


def test_enabled_false_does_not_prepare_or_upload(audio_context):
    make, clients, _, _ = audio_context
    context = make(enabled=False)
    assert context.for_request().file_uri is None
    assert not clients


@pytest.mark.parametrize("options", [{"enabled": "false"}, {"compress_long_audio": "false"},
                                     {"cache_enabled": "false"}, {"unknown": 1},
                                     {"upload_timeout_seconds": -1}])
def test_invalid_options_fail_before_network(audio_context, options):
    make, clients, _, _ = audio_context
    with pytest.raises(ProviderError):
        make(**options)
    assert not clients


def test_cached_source_context_is_read_only_and_sdk_compatible(audio_context):
    sdk_types = pytest.importorskip("google.genai.types")
    make, clients, _, _ = audio_context
    context = make()
    context.source_context = '[{"cue_id": 1, "text": "source dialogue"}]'
    context.for_request()
    config = clients[0].caches.created[0][1]
    parsed = sdk_types.CreateCachedContentConfig.model_validate(config)
    assert parsed.ttl is None
    assert parsed.expire_time == datetime(2026, 9, 10, tzinfo=timezone.utc) + timedelta(seconds=900)
    assert parsed.contents[0].parts[1].file_data.file_uri == "https://example.test/owned"
    assert "source dialogue" in parsed.contents[0].parts[0].text
    context.close()


def test_cache_renewal_failure_falls_back_without_using_stale_cache(audio_context):
    make, clients, _, clock = audio_context
    context = make(cache_ttl_seconds=120)
    context.for_request()

    def fail(**kwargs):
        raise RuntimeError("expired")

    clients[0].caches.update = fail
    clock[0] = 70
    result = context.for_request()
    assert result.cached_content is None and result.file_uri
    assert "cache_renewal_failed" in context.report()["warnings"]
    assert clients[0].caches.deleted == ["cachedContents/owned"]
    context.close()


def test_missing_file_uri_still_deletes_owned_upload(audio_context):
    make, clients, _, _ = audio_context
    context = make()
    context._open_client()
    clients[0].files.upload = lambda **kwargs: SimpleNamespace(name="files/owned", uri=None, state="ACTIVE")
    with pytest.raises(ProviderError):
        context.for_request()
    assert clients[0].files.deleted == ["files/owned"]


def test_lost_cache_create_response_recovers_and_deletes_only_job_owned_cache(audio_context):
    make, clients, _, _ = audio_context
    context = make()
    context._open_client()
    owned = []

    def create(**kwargs):
        owned.append(SimpleNamespace(name="cachedContents/lost-owned", display_name=kwargs["config"]["display_name"]))
        raise TimeoutError("response lost")

    clients[0].caches.create = create
    clients[0].caches.list = lambda **kwargs: [SimpleNamespace(name="cachedContents/user-owned", display_name="someone-else"), *owned]
    assert context.for_request().file_uri
    assert clients[0].caches.deleted == ["cachedContents/lost-owned"]
    assert "cache_creation_unconfirmed" not in context.report()["warnings"]
    context.close()
    assert context.report()["cleanup_status"] == "complete"


def test_lost_upload_response_recovers_and_deletes_only_job_owned_file(audio_context):
    make, clients, _, _ = audio_context
    context = make()
    context._open_client()
    owned = []

    def upload(**kwargs):
        owned.append(SimpleNamespace(name="files/lost-owned", display_name=kwargs["config"]["display_name"]))
        raise TimeoutError("response lost")

    clients[0].files.upload = upload
    clients[0].files.list = lambda **kwargs: [SimpleNamespace(name="files/user-owned", display_name="someone-else"), *owned]
    with pytest.raises(ProviderError):
        context.for_request()
    assert clients[0].files.deleted == ["files/lost-owned"]
    assert context.report()["cleanup_status"] == "complete"


def test_unknown_creation_that_cannot_be_recovered_is_reported_incomplete(audio_context):
    make, clients, _, _ = audio_context
    context = make()
    context._open_client()
    clients[0].caches.error = TimeoutError("response lost")
    clients[0].caches.list = lambda **kwargs: []
    assert context.for_request().file_uri
    context.close()
    assert "cache_creation_unconfirmed" in context.report()["warnings"]
    assert context.report()["cleanup_status"] == "incomplete"
    assert context.report()["cache_storage_token_seconds_reserved"] > 9600 * 900


def test_context_provenance_detects_changed_source_before_generation(audio_context):
    make, clients, path, _ = audio_context
    context = make()
    context._open_client()
    upload = clients[0].files.upload

    def changed_during_upload(**kwargs):
        path.write_bytes(b"modified original")
        return upload(**kwargs)

    clients[0].files.upload = changed_during_upload
    with pytest.raises(ProviderError):
        context.for_request()
    assert clients[0].files.deleted == ["files/owned"]
    assert not clients[0].caches.created


@pytest.mark.parametrize("token_count", [None, "invalid", float("nan"), -5, True])
def test_missing_cache_usage_cannot_break_the_lease_or_cleanup(audio_context, token_count):
    make, clients, _, _ = audio_context
    context = make()
    context._open_client()
    context.source_context = "source text" * 1000
    clients[0].caches.create = lambda **kwargs: SimpleNamespace(
        name="cachedContents/owned", usage_metadata=SimpleNamespace(total_token_count=token_count))
    assert context.for_request().cached_content == "cachedContents/owned"
    assert context.report()["cache_token_count"] > 9600 + 10000
    assert "cache_usage_estimated" in context.report()["warnings"]
    context.close()
    assert context.report()["cleanup_status"] == "complete"
    assert clients[0].caches.deleted == ["cachedContents/owned"]


def test_ambiguous_cache_creation_reserves_text_prefix_as_well_as_audio(audio_context):
    make, clients, _, _ = audio_context
    context = make()
    context._open_client()
    context.source_context = "source text" * 1000
    clients[0].caches.error = TimeoutError("response lost")
    assert context.for_request().file_uri
    report = context.report()
    assert report["cache_create_input_tokens_reserved"] > 9600 + 10000
    assert report["cache_storage_token_seconds_reserved"] == report["cache_create_input_tokens_reserved"] * 900
    context.close()


def test_four_concurrent_requests_share_one_upload_and_cache(audio_context):
    make, clients, _, _ = audio_context
    context = make()
    barrier = Barrier(4)

    def request(_):
        barrier.wait(timeout=5)
        lease = context.acquire_request()
        try:
            barrier.wait(timeout=5)
            return lease.cached_content
        finally:
            context.release_request()

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(request, range(4))) == ["cachedContents/owned"] * 4
    assert len(clients) == 1
    assert len(clients[0].files.uploads) == len(clients[0].caches.created) == 1
    assert context.report()["in_flight_requests"] == 0
    context.close()
    assert clients[0].caches.deleted == ["cachedContents/owned"]
    assert clients[0].files.deleted == ["files/owned"]


def test_open_circuit_defers_cleanup_until_all_in_flight_requests_finish(audio_context):
    make, clients, _, _ = audio_context
    context = make()
    leases = [context.acquire_request() for _ in range(4)]
    context.record_generation_result(success=False, cached=True)
    context.record_generation_result(success=False, cached=True)
    assert context.report()["cleanup_status"] == "pending"
    assert clients[0].caches.deleted == clients[0].files.deleted == []
    with pytest.raises(ProviderError):
        context.acquire_request()
    for _ in leases[:-1]:
        context.release_request()
        assert clients[0].caches.deleted == []
    context.release_request()
    assert clients[0].caches.deleted == ["cachedContents/owned"]
    assert clients[0].files.deleted == ["files/owned"]
    assert clients[0].closed
    assert context.report()["cleanup_status"] == "complete"


def test_uncached_budget_reservations_are_atomic_across_workers(audio_context):
    make, clients, _, _ = audio_context
    context = make(duration=60, max_uncached_audio_tokens=3840)
    barrier = Barrier(4)

    def request(_):
        barrier.wait(timeout=5)
        try:
            context.acquire_request()
        except ProviderError:
            return False
        else:
            context.release_request()
            return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        accepted = list(executor.map(request, range(4)))
    assert sum(accepted) == 2
    assert context.report()["uncached_audio_tokens_reserved"] == 3840
    assert context.report()["in_flight_requests"] == 0
    assert len(clients[0].files.uploads) == 1
    context.close()


@pytest.mark.parametrize("resource_kind", ["files", "caches"])
def test_interrupted_resource_creation_cleans_owned_objects_and_preserves_interrupt(audio_context, resource_kind):
    make, clients, _, _ = audio_context
    context = make()
    context._open_client()
    service = getattr(clients[0], resource_kind)
    owned = []
    name = "files/interrupted" if resource_kind == "files" else "cachedContents/interrupted"

    def interrupted(**kwargs):
        owned.append(SimpleNamespace(name=name, display_name=kwargs["config"]["display_name"]))
        raise KeyboardInterrupt()

    setattr(service, "upload" if resource_kind == "files" else "create", interrupted)
    service.list = lambda **kwargs: owned
    with pytest.raises(KeyboardInterrupt):
        context.acquire_request()
    assert name in service.deleted
    assert clients[0].closed
    assert context.report()["in_flight_requests"] == 0
    assert context.report()["cleanup_status"] == "complete"


def test_failed_cache_renewal_cannot_delete_cache_used_by_an_active_request(audio_context):
    make, clients, _, clock = audio_context
    context = make()
    context.for_request()
    clock[0] = 800
    assert context.acquire_request().cached_content

    def failed_update(**kwargs):
        raise TimeoutError("renewal unavailable")

    clients[0].caches.update = failed_update
    clock[0] = 845
    with pytest.raises(ProviderError, match="cache lease"):
        context.acquire_request()
    assert clients[0].caches.deleted == clients[0].files.deleted == []
    assert context.report()["cleanup_status"] == "pending"
    assert context.report()["in_flight_requests"] == 1
    context.release_request()
    assert clients[0].caches.deleted == ["cachedContents/owned"]
    assert clients[0].files.deleted == ["files/owned"]
    assert context.report()["cleanup_status"] == "complete"


def test_slow_successful_renewal_cannot_grant_a_lease_after_job_deadline(audio_context):
    make, clients, _, clock = audio_context
    context = make(cache_ttl_seconds=60, max_lifetime_seconds=100)
    context.for_request()
    clock[0] = 45
    updates = []

    def slow_update(**kwargs):
        updates.append(kwargs)
        clock[0] = 105

    clients[0].caches.update = slow_update
    with pytest.raises(ProviderError, match="job lifetime"):
        context.acquire_request()
    assert context.report()["cached_requests"] == 1
    assert context.report()["in_flight_requests"] == 0
    assert context.report()["cleanup_status"] == "complete"
    assert updates[0]["config"] == {"expire_time": datetime(2026, 9, 10, tzinfo=timezone.utc) + timedelta(seconds=100)}


def test_slow_successful_renewal_cannot_grant_an_already_expired_cache(audio_context):
    make, clients, _, clock = audio_context
    context = make(cache_ttl_seconds=60, max_lifetime_seconds=1000)
    context.for_request()
    clock[0] = 45
    clients[0].caches.update = lambda **kwargs: clock.__setitem__(0, 110)
    with pytest.raises(ProviderError, match="cache lease expired"):
        context.acquire_request()
    assert "cache_lease_expired" in context.report()["warnings"]
    assert context.report()["cached_requests"] == 1
    assert context.report()["in_flight_requests"] == 0
    assert context.report()["cleanup_status"] == "complete"
