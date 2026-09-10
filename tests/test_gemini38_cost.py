from __future__ import annotations

import copy
from datetime import date
from types import SimpleNamespace

import pytest

import dubsync.cost as cost


@pytest.fixture(autouse=True)
def introductory_prices(monkeypatch):
    monkeypatch.setattr(cost, "_utc_today", lambda: date(2026, 9, 10))


def generation_usage(cached_tokens=96_000):
    return {
        "usage_metadata": {
            "prompt_token_count": 100_000,
            "cached_content_token_count": cached_tokens,
            "candidates_token_count": 400,
            "thoughts_token_count": 600,
        }
    }


def test_gemini38_prices_follow_official_date_boundary(monkeypatch):
    for model in ("gemini-3.8-flash", "models/gemini-3.8-flash"):
        monkeypatch.setattr(cost, "_utc_today", lambda: date(2026, 12, 31))
        assert cost.llm_token_prices("gemini", model, {}) == (0.75, 3.75)
        monkeypatch.setattr(cost, "_utc_today", lambda: date(2027, 1, 1))
        assert cost.llm_token_prices("gemini", model, {}) == (1.5, 7.5)


def test_gemini38_explicit_token_price_override_is_preserved():
    assert cost.llm_token_prices(
        "gemini", "gemini-3.8-flash", {"pricing": {"input_per_million": 2, "output_per_million": 8}}
    ) == (2.0, 8.0)


def test_cached_usage_preserves_total_prompt_and_adds_thinking_to_output():
    usage = cost.token_usage_from_response(generation_usage())
    assert usage is not None
    assert usage.input_tokens == 100_000
    assert usage.cached_input_tokens == 96_000
    assert usage.output_tokens == 1_000
    assert cost.TokenUsage(input_tokens=10, output_tokens=2).cached_input_tokens == 0


def test_cached_usage_accepts_sdk_objects_and_rest_field_names():
    sdk_response = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=100_000, cached_content_token_count=96_000,
        candidates_token_count=400, thoughts_token_count=600,
    ))
    rest_response = {"usageMetadata": {
        "promptTokenCount": "100000", "cachedContentTokenCount": "96000",
        "candidatesTokenCount": 400, "thoughtsTokenCount": 600,
    }}
    assert cost.token_usage_from_response(sdk_response) == cost.token_usage_from_response(rest_response)
    assert cost.token_usage_from_response(rest_response).cached_input_tokens == 96_000


@pytest.mark.parametrize("invalid", [-1, "unknown", float("nan"), float("inf"), True, 100_001, 1.5])
def test_invalid_cached_counts_cannot_discount_the_prompt(invalid):
    usage = cost.token_usage_from_response(generation_usage(invalid))
    assert usage is not None
    assert usage.cached_input_tokens == 0


@pytest.mark.parametrize("invalid", [-1, "unknown", float("nan"), float("inf"), True, 1.5])
def test_invalid_required_token_counts_are_unmetered_without_crashing(invalid):
    response = generation_usage()
    response["usage_metadata"]["prompt_token_count"] = invalid
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, response) == "usage metadata unavailable"
    assert meter.items == []


def test_cached_read_cost_does_not_double_bill_cached_prompt_tokens():
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, generation_usage()) is None
    assert meter.total_usd == 0.01395
    assert meter.items[0].units == {
        "input_tokens": 100_000.0, "output_tokens": 1_000.0,
        "cached_input_tokens": 96_000.0, "uncached_input_tokens": 4_000.0,
    }


def test_cached_read_rates_change_on_january_first(monkeypatch):
    monkeypatch.setattr(cost, "_utc_today", lambda: date(2027, 1, 1))
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "models/gemini-3.8-flash", {}, generation_usage()) is None
    assert meter.total_usd == 0.0279


@pytest.mark.parametrize("response", [
    {"usage_metadata": {"input_token_count": 1_000, "output_token_count": 200}},
    {"usage": {"prompt_tokens": 1_000, "completion_tokens": 200}},
])
def test_uncached_cost_and_item_shape_remain_backward_compatible(response):
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, response) is None
    assert meter.total_usd == 0.0015
    assert meter.items[0].kind == "tokens"
    assert meter.items[0].units == {"input_tokens": 1_000.0, "output_tokens": 200.0}


def test_unknown_model_needs_prices_and_never_receives_an_invented_cache_discount():
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "future-model", {}, generation_usage()) == "token pricing unavailable"
    assert meter.items == []
    config = {"input_per_million": 2, "output_per_million": 8}
    assert cost.record_llm_usage(meter, "gemini", "future-model", config, generation_usage()) is None
    assert meter.total_usd == 0.208


def test_custom_prices_require_an_explicit_custom_cache_discount():
    config = {"pricing": {"input_per_million": 2, "output_per_million": 8}}
    meter = cost.CostMeter()
    cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", config, generation_usage())
    assert meter.total_usd == 0.208
    config["pricing"]["cached_input_per_million"] = 0.2
    discounted_meter = cost.CostMeter()
    cost.record_llm_usage(discounted_meter, "gemini", "gemini-3.8-flash", config, generation_usage())
    assert discounted_meter.total_usd == 0.0352


def test_non_gemini_provider_does_not_use_native_gemini_cache_semantics():
    meter = cost.CostMeter()
    cost.record_llm_usage(meter, "openrouter", "gemini-3.8-flash", {"cached_input_per_million": 0}, generation_usage())
    assert meter.total_usd == 0.07875
    assert "cached_input_tokens" not in meter.items[0].units


def test_cache_context_records_creation_once_and_actual_storage_as_estimates():
    meter = cost.CostMeter()
    report = {
        "cache_token_count": 96_000,
        "cache_create_input_tokens_reserved": 96_000,
        "cache_storage_token_seconds": 96_000 * 180,
        "cache_renewals": 3,
        "cached_requests": 5,
        "uncached_requests": 2,
        "uncached_audio_tokens_reserved": 192_000,
    }
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, report) is None
    assert [item.kind for item in meter.items] == ["cache_create_estimate", "cache_storage_estimate"]
    assert meter.items[0].usd == 0.072
    assert meter.items[1].usd == 0.0024
    assert meter.items[1].units["token_seconds"] == 96_000 * 180
    assert meter.total_usd == 0.0744
    assert report["cache_create_input_tokens_reserved"] == 96_000


def test_failed_cleanup_uses_total_reserved_storage_bound_without_adding_it_twice():
    meter = cost.CostMeter()
    report = {
        "cache_create_input_tokens_reserved": 96_000,
        "cache_storage_token_seconds": 96_000 * 180,
        "cache_storage_token_seconds_reserved": 96_000 * 900,
        "cleanup_status": "cache_delete_failed",
    }
    cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, report)
    assert meter.items[1].usd == 0.012
    assert meter.total_usd == 0.084


def test_cache_context_storage_prices_follow_january_boundary(monkeypatch):
    monkeypatch.setattr(cost, "_utc_today", lambda: date(2027, 1, 1))
    meter = cost.CostMeter()
    report = {"cache_create_input_tokens_reserved": 96_000, "cache_storage_token_seconds": 96_000 * 900}
    cost.record_gemini_context_cost(meter, "models/gemini-3.8-flash", {}, report)
    assert meter.total_usd == 0.168


def test_cache_context_supports_explicit_storage_prices_for_unknown_models():
    meter = cost.CostMeter()
    report = {"cache_create_input_tokens_reserved": 1_000_000, "cache_storage_token_seconds": 3_600_000_000}
    config = {"cost": {"input_per_million": 2, "output_per_million": 8, "cache_storage_per_million_token_hour": 4}}
    assert cost.record_gemini_context_cost(meter, "future-model", config, report) is None
    assert meter.total_usd == 6


def test_cache_context_without_activity_never_creates_charges():
    meter = cost.CostMeter()
    assert cost.record_gemini_context_cost(meter, "future-model", {}, {"status": "disabled"}) is None
    assert meter.items == []


def test_cache_context_with_unknown_prices_is_reported_unmetered():
    meter = cost.CostMeter()
    report = {"cache_create_input_tokens_reserved": 96_000, "cache_storage_token_seconds": 96_000 * 900}
    assert cost.record_gemini_context_cost(meter, "future-model", {}, report) == "cache context pricing unavailable"
    assert meter.items == []


@pytest.mark.parametrize("field", ["cache_create_input_tokens_reserved", "cache_storage_token_seconds", "cache_storage_token_seconds_reserved"])
@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), "unknown", True])
def test_cache_context_rejects_invalid_usage_metadata(field, invalid):
    meter = cost.CostMeter()
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, {field: invalid}) == "cache context usage metadata invalid"
    assert meter.items == []


@pytest.mark.parametrize("field", ["input_per_million", "cached_input_per_million"])
@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), "unknown", True])
def test_invalid_generation_prices_are_unmetered_without_crashing(field, invalid):
    meter = cost.CostMeter()
    config = {"input_per_million": 2, "output_per_million": 8, field: invalid}
    assert cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", config, generation_usage()) == "token pricing invalid"
    assert meter.items == []


def test_invalid_cache_storage_price_is_unmetered_without_crashing():
    meter = cost.CostMeter()
    config = {"cache_storage_per_million_token_hour": float("nan")}
    report = {"cache_storage_token_seconds": 96_000 * 900}
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", config, report) == "cache context pricing invalid"
    assert meter.items == []


def test_failed_uri_requests_reserve_only_unreported_audio_input_as_uncertain_estimate():
    meter = cost.CostMeter()
    report = {
        "uncached_audio_tokens_reserved": 288_000,
        "unreported_uncached_audio_tokens_reserved": 192_000,
    }
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, report) is None
    assert len(meter.items) == 1
    assert meter.items[0].kind == "uncertain_audio_input_estimate"
    assert meter.items[0].units == {"input_tokens": 192_000.0}
    assert meter.total_usd == 0.144


def test_failed_uri_reserve_does_not_duplicate_successful_generation_usage():
    meter = cost.CostMeter()
    cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, generation_usage(0))
    report = {
        "uncached_audio_tokens_reserved": 192_000,
        "unreported_uncached_audio_tokens_reserved": 96_000,
    }
    cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, report)
    assert [item.kind for item in meter.items] == ["tokens", "uncertain_audio_input_estimate"]
    assert meter.total_usd == 0.15075


def test_successful_uri_reservations_without_unreported_requests_are_not_billed_twice():
    meter = cost.CostMeter()
    assert cost.record_gemini_context_cost(
        meter, "gemini-3.8-flash", {}, {"uncached_audio_tokens_reserved": 96_000}
    ) is None
    assert meter.items == []


def test_failed_uri_reserve_requires_known_or_explicit_input_prices():
    meter = cost.CostMeter()
    report = {"unreported_uncached_audio_tokens_reserved": 96_000}
    assert cost.record_gemini_context_cost(meter, "future-model", {}, report) == "cache context pricing unavailable"
    assert meter.items == []
    config = {"input_per_million": 2, "output_per_million": 8}
    assert cost.record_gemini_context_cost(meter, "future-model", config, report) is None
    assert meter.total_usd == 0.192


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), "unknown", True, 1.5])
def test_failed_uri_reserve_rejects_invalid_token_counts(invalid):
    meter = cost.CostMeter()
    report = {"unreported_uncached_audio_tokens_reserved": invalid}
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, report) == "cache context usage metadata invalid"
    assert meter.items == []


def test_failed_cached_requests_add_uncertain_cached_read_estimate_separately():
    meter = cost.CostMeter()
    report = {
        "unreported_uncached_audio_tokens_reserved": 96_000,
        "unreported_cached_audio_tokens_reserved": 192_000,
    }
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, report) is None
    assert [item.kind for item in meter.items] == [
        "uncertain_audio_input_estimate", "uncertain_cached_input_estimate",
    ]
    assert meter.items[1].units == {"cached_input_tokens": 192_000.0}
    assert meter.items[1].usd == 0.0144
    assert meter.total_usd == 0.0864


def test_failed_cached_request_does_not_duplicate_successful_generation_usage():
    meter = cost.CostMeter()
    cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, generation_usage())
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, {
        "unreported_cached_audio_tokens_reserved": 96_000,
    }) is None
    assert meter.total_usd == 0.02115


def test_failed_cached_read_estimates_follow_january_pricing(monkeypatch):
    monkeypatch.setattr(cost, "_utc_today", lambda: date(2027, 1, 1))
    meter = cost.CostMeter()
    assert cost.record_gemini_context_cost(meter, "models/gemini-3.8-flash", {}, {
        "unreported_cached_audio_tokens_reserved": 96_000,
    }) is None
    assert meter.total_usd == 0.0144


def test_failed_cached_read_uses_explicit_discount_or_normal_custom_input_price():
    report = {"unreported_cached_audio_tokens_reserved": 96_000}
    config = {"input_per_million": 2, "output_per_million": 8}
    meter = cost.CostMeter()
    assert cost.record_gemini_context_cost(meter, "future-model", {}, report) == "cache context pricing unavailable"
    assert meter.items == []
    assert cost.record_gemini_context_cost(meter, "future-model", config, report) is None
    assert meter.total_usd == 0.192
    config["cached_input_per_million"] = 0.2
    discounted_meter = cost.CostMeter()
    assert cost.record_gemini_context_cost(discounted_meter, "future-model", config, report) is None
    assert discounted_meter.total_usd == 0.0192


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), "unknown", True, 1.5])
def test_failed_cached_read_rejects_invalid_token_reserves(invalid):
    meter = cost.CostMeter()
    report = {"unreported_cached_audio_tokens_reserved": invalid}
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {}, report) == "cache context usage metadata invalid"
    assert meter.items == []


def test_failed_cached_read_invalid_tariff_is_flagged_without_partial_accounting():
    meter = cost.CostMeter()
    report = {
        "unreported_uncached_audio_tokens_reserved": 96_000,
        "unreported_cached_audio_tokens_reserved": 96_000,
    }
    assert cost.record_gemini_context_cost(meter, "gemini-3.8-flash", {
        "cached_input_per_million": -1,
    }, report) == "cache context pricing invalid"
    assert meter.items == []


def live_inconsistent_cache_usage():
    # Observed Gemini 3.8 cached-single-case probe, 2026-09-10. Its cached
    # audio count exceeds prompt audio despite the documented inclusive total.
    return {"usage_metadata": {
        "prompt_token_count": 143_548,
        "cached_content_token_count": 163_099,
        "candidates_token_count": 119,
        "thoughts_token_count": 2_679,
        "total_token_count": 146_346,
        "prompt_tokens_details": [
            {"modality": "TEXT", "token_count": 69_301},
            {"modality": "AUDIO", "token_count": 74_247},
        ],
        "cache_tokens_details": [
            {"modality": "TEXT", "token_count": 68_241},
            {"modality": "AUDIO", "token_count": 94_858},
        ],
    }}


def test_real_inconsistent_cache_counts_remain_metered_as_explicit_uncertain_estimate():
    response = live_inconsistent_cache_usage()
    original = copy.deepcopy(response)
    meter = cost.CostMeter()
    # None means accounted: returning an unmetered reason here would falsely
    # tell the caller that this amount was excluded from the total.
    assert cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, response) is None
    assert meter.total_usd == 0.118154
    assert meter.items[0].kind == "tokens_cache_metadata_estimate"
    assert meter.items[0].units == {
        "input_tokens": 143_548.0,
        "output_tokens": 2_798.0,
        "reported_cached_input_tokens": 163_099.0,
    }
    assert meter.as_dict()["items"][0]["units"]["reported_cached_input_tokens"] == 163_099.0
    assert response == original


def test_inconsistent_cache_usage_keeps_reported_count_separate_from_applied_discount():
    usage = cost.token_usage_from_response(live_inconsistent_cache_usage())
    assert usage.cached_input_tokens == 0
    assert usage.reported_cached_input_tokens == 163_099
    assert usage.cache_metadata_inconsistent is True


def test_rest_cache_count_inconsistency_is_also_marked():
    response = {"usageMetadata": {
        "promptTokenCount": 10, "cachedContentTokenCount": 11,
        "candidatesTokenCount": 2, "thoughtsTokenCount": 3,
    }}
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, response) is None
    assert meter.items[0].kind == "tokens_cache_metadata_estimate"
    assert meter.items[0].units["reported_cached_input_tokens"] == 11.0


def test_non_gemini_provider_does_not_claim_native_cache_metadata_uncertainty():
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "openrouter", "gemini-3.8-flash", {}, live_inconsistent_cache_usage()) is None
    assert meter.items[0].kind == "tokens"
    assert meter.items[0].units == {"input_tokens": 143_548.0, "output_tokens": 2_798.0}


def test_inconsistent_cache_metadata_still_requires_known_prices_before_accounting():
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "future-model", {}, live_inconsistent_cache_usage()) == "token pricing unavailable"
    assert meter.items == []


@pytest.mark.parametrize("invalid", [-1, "unknown", float("nan"), float("inf"), True, 1.5])
def test_invalid_cache_metadata_is_marked_without_claiming_it_was_unmetered(invalid):
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, generation_usage(invalid)) is None
    assert meter.items[0].kind == "tokens_cache_metadata_estimate"
    assert meter.total_usd == 0.07875


def test_absent_cache_count_has_no_native_cache_uncertainty_marker():
    response = generation_usage(None)
    meter = cost.CostMeter()
    assert cost.record_llm_usage(meter, "gemini", "gemini-3.8-flash", {}, response) is None
    assert meter.items[0].kind == "tokens"
    assert cost.token_usage_from_response(response).cache_metadata_inconsistent is False
