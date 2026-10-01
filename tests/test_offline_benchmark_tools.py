from __future__ import annotations

import importlib.util
import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubsync.models import Cue


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f"benchmark_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def corpus(tmp_path):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"".join(struct.pack("<h", int(9000 * math.sin(i * math.tau * 440 / 16000))
                                                 if 3200 <= i < 12800 else 0) for i in range(16000)))
    (tmp_path / "source.srt").write_text("1\n00:00:00,100 --> 00:00:00,900\nHello world.\n", encoding="utf-8")
    (tmp_path / "words.json").write_text(json.dumps({"words": [
        {"text": "Hello", "start": 0.2, "end": 0.45},
        {"text": "world.", "start": 0.5, "end": 0.8},
    ]}), encoding="utf-8")
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"episodes": [{
        "id": "clip", "source_srt": {"path": "source.srt"},
        "normalized_16k_candidates": [{"path": "audio.wav", "is_full_episode": True}],
        "asr": {"mai": [{"id": "fixture", "coverage": "full", "fixture_path": "words.json"}]},
        "stage_dirs": {},
        "offline_replay": {"commands": [{"model": "mai", "transcript": "fixture", "saved": False}]},
    }]}), encoding="utf-8")
    return manifest


def test_replay_is_portable_and_never_replaces_existing_evidence(corpus):
    out = corpus.parent / "result"
    command = [sys.executable, str(SCRIPTS / "replay_offline.py"), "--manifest", str(corpus),
               "--episode", "clip", "--out", str(out)]
    first = subprocess.run(command, cwd=corpus.parent, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    run = out / "clip.mai.nollm"
    evidence = json.loads((run / "replay-evidence.json").read_text(encoding="utf-8"))
    assert evidence["cues"] == 1
    assert evidence["model"] == "mai"
    timing = load_tool("timing_vs_audio").main(out, None)
    assert timing["clip.mai.nollm"]["first_word_cut"]["count"] == 0
    assert timing["clip.mai.nollm"]["overlap_pairs"] == 0
    before = (run / "output.srt").read_bytes()
    second = subprocess.run(command, cwd=corpus.parent, capture_output=True, text=True)
    assert second.returncode != 0
    assert (run / "output.srt").read_bytes() == before


def test_corpus_cli_reports_failed_replays_with_nonzero_exit(corpus):
    (corpus.parent / "words.json").unlink()
    result = subprocess.run([sys.executable, str(SCRIPTS / "bench_corpus.py"), "--manifest", str(corpus),
                             "--out", str(corpus.parent / "bad"), "--jobs", "1"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    summary = json.loads((corpus.parent / "bad" / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["failures"]) == 1


def test_golden_distinguishes_text_errors_and_human_changed_edges():
    tool = load_tool("golden_bench")
    source = [Cue(index=1, start_ms=0, end_ms=1000, lines=["alpha beta"])]
    golden = [Cue(index=1, start_ms=100, end_ms=1000, lines=["alpha beta"])]
    swapped = [Cue(index=1, start_ms=100, end_ms=1000, lines=["beta alpha"])]
    result = tool.compare(source, golden, swapped)
    assert result["text"]["wer_vs_golden"] > 0
    assert result["order"]["confirmed_transpositions"] == 1
    subset = tool.human_edit_subset(golden, source, {"fixed": golden})
    assert subset["edited_edges"] == {"start": 1, "end": 0}
    assert subset["outputs"]["fixed"]["start"]["edited_stats"]["mae"] == 0


def test_golden_rejects_missing_output_and_distant_duplicate_is_not_transposition(tmp_path):
    tool = load_tool("golden_bench")
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    assert tool.confirmed_transpositions(tokens, tokens[1:] + ["alpha"]) == []
    source = tmp_path / "s.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        tool.run_suite({"clip": {"source": source, "golden": source,
                                "outputs": {"missing": tmp_path / "missing.srt"}}}, 2)


def test_timing_envelope_is_streamed_and_measures_fixture_audio(corpus):
    tool = load_tool("timing_vs_audio")
    db = tool.envelope(corpus.parent / "audio.wav")
    starts, ends = tool.bursts(db)
    assert len(starts) == len(ends) == 1
    assert starts[0] == pytest.approx(0.19)
    assert ends[0] == pytest.approx(0.8)
    assert tool.pcts([-.02, 0, .02]) == {"n": 3, "p05": -18, "p25": -10, "p50": 0,
                                        "p75": 10, "p95": 18, "mean_abs": 13,
                                        "within40": 100.0, "within80": 100.0}


def test_saved_verdicts_report_exact_loose_and_ambiguous_coverage(tmp_path):
    tool = load_tool("replay_offline")
    spans = [{"case_id": name, "srt_text": source, "asr_text": heard,
              "cue_ids": [index], "asr_word_indices": [index]}
             for index, (name, source, heard) in enumerate([
                 ("exact", "original", "heard"), ("loose", "source", "speech"),
                 ("repeat1", "Vamos", "Vai"), ("repeat2", "Vamos", "Vai")])]
    (tmp_path / "align.json").write_text(json.dumps({"divergence_spans": spans}), encoding="utf-8")
    decisions = [{"case_id": span["case_id"], "verdict": "keep_srt" if index == 3 else "use_audio",
                  "final_text": span["srt_text"] if index == 3 else span["asr_text"], "confidence": .9}
                 for index, span in enumerate(spans)]
    (tmp_path / "adjudicate.json").write_text(json.dumps({"decisions": decisions}), encoding="utf-8")
    incoming = [spans[0], {**spans[1], "case_id": "new-loose", "cue_ids": [20]},
                {**spans[2], "case_id": "ambiguous", "cue_ids": [30]}]
    payloads = [SimpleNamespace(**span, model_dump=lambda value=span: value) for span in incoming]
    adapter = tool.SavedDecisionAdapter(tmp_path)
    actual = adapter.adjudicate(payloads)
    assert adapter.matched == ["exact"]
    assert adapter.matched_loose == ["new-loose"]
    assert adapter.held == ["ambiguous"]
    assert actual[-1]["confidence"] == 0
    assert actual[-1]["final_text"] == "Vamos"


def test_added_reaction_does_not_turn_unchanged_word_into_confirmed_transposition():
    tool = load_tool("golden_bench")
    golden = "aquela garota ali olha luan nian".split()
    before = "aquela olha luan nian".split()
    after = "aquela olha ha luan nian".split()
    assert tool.confirmed_transpositions(golden, before) == []
    assert tool.confirmed_transpositions(golden, after) == []


@pytest.mark.parametrize("different", [{"confidence": 0}, {"evidence": "heard_unclear", "heard_text": "Stay"}, {"speaker": "other"}])
def test_loose_replay_does_not_borrow_different_certainty(tmp_path, different):
    tool = load_tool("replay_offline")
    spans = [dict(case_id=str(i), srt_text="Stay", asr_text="Wait", cue_ids=[i], asr_word_indices=[i]) for i in (1, 2)]
    decisions = [dict(case_id=str(i), verdict="keep_srt", final_text="Stay", confidence=1, reason="saved") for i in (1, 2)]
    decisions[1].update(different)
    (tmp_path / "align.json").write_text(json.dumps({"divergence_spans": spans}), encoding="utf-8")
    (tmp_path / "adjudicate.json").write_text(json.dumps({"decisions": decisions}), encoding="utf-8")
    incoming = {**spans[0], "case_id": "new", "cue_ids": [3]}
    adapter = tool.SavedDecisionAdapter(tmp_path)
    result = adapter.adjudicate([SimpleNamespace(**incoming, model_dump=lambda: incoming)])
    assert adapter.held == ["new"]
    assert result[0]["confidence"] == 0
