import json

import pytest
import yaml

from dubsync import pipeline


@pytest.mark.parametrize(("selected", "configured", "short", "expected"), [
    (None, None, False, "pt"), ("auto", None, False, "pt"),
    ("en", None, False, "en"), (None, "fr", False, "fr"),
    (None, None, True, None),
])
def test_sync_language_hint_is_bound_before_asr_and_respects_selection(
    tmp_path, monkeypatch, selected, configured, short, expected,
):
    text = ("Não." if short else "Você não está aqui porque eu estou falando com vocês. Então também pode fazer alguma coisa para ela.")
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:10,000\n" + text + "\n", encoding="utf-8")
    fixture = tmp_path / "words.json"
    fixture.write_text(json.dumps({"words": [dict(text=word, start=i*.3, end=i*.3+.2)
                                              for i, word in enumerate(text.split())]}), encoding="utf-8")
    config = {"asr": {"fixture_path": str(fixture)}}
    if configured:
        config["asr"]["language_code"] = configured
    providers = tmp_path / "providers.yaml"
    providers.write_text(yaml.safe_dump(config), encoding="utf-8")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")
    seen = []
    original = pipeline.adapter_from_config
    def adapter(configuration, **kwargs):
        seen.append(configuration["asr"].get("language_code"))
        return original(configuration, **kwargs)
    monkeypatch.setattr(pipeline, "adapter_from_config", adapter)
    result = pipeline.sync_episode(source, audio, tmp_path / "out.srt", tmp_path / "work",
                                   providers_path=providers, no_llm=True, language=selected)
    assert seen == [expected]
    inferred = [flag for flag in result.report["flags"] if flag["kind"] == "source_language_inferred"]
    assert bool(inferred) is (expected == "pt")
