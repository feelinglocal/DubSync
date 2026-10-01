"""Replay one cached corpus transcript without network access or provider credentials.

Paths in --manifest resolve against its repo_root (when given), otherwise its directory.
The output run directory must not already exist. Saved decisions are fixture evidence,
not fresh provider quality; exact, text-only and unmatched counts are reported separately.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("DUBSYNC_SRC") or str(ROOT / "src"))

import yaml


def identity(span):
    return (span["srt_text"], span["asr_text"], tuple(span["cue_ids"]), tuple(span.get("asr_word_indices", [])))


def loose_identity(span):
    return tuple(" ".join(span[k].split()).casefold() for k in ("srt_text", "asr_text"))


class SavedDecisionAdapter:
    def __init__(self, stage: Path, *, allow_loose: bool = True):
        old = json.loads((stage / "align.json").read_text(encoding="utf-8"))
        decisions = {d["case_id"]: d for d in json.loads(
            (stage / "adjudicate.json").read_text(encoding="utf-8"))["decisions"]}
        self.exact, self.loose = {}, collections.defaultdict(list)
        for span in old["divergence_spans"]:
            if span["case_id"] in decisions:
                self.exact[identity(span)] = decisions[span["case_id"]]
                self.loose[loose_identity(span)].append(decisions[span["case_id"]])
        self.allow_loose = allow_loose
        self.matched, self.matched_loose, self.held = [], [], []

    def adjudicate(self, spans):
        results = []
        for span in spans:
            data = span.model_dump()
            saved = self.exact.get(identity(data))
            if saved is not None:
                self.matched.append(span.case_id)
            elif self.allow_loose and self.loose.get(loose_identity(data)):
                candidates = self.loose[loose_identity(data)]
                # Ambiguous repeated text must not borrow an unrelated verdict.
                verdicts = {tuple(d.get(key) for key in (
                    "verdict", "final_text", "confidence", "evidence", "heard_text", "speaker", "character",
                )) for d in candidates}
                if len(verdicts) == 1:
                    saved = candidates[0]
                    self.matched_loose.append(span.case_id)
            if saved is None:
                self.held.append(span.case_id)
                saved = dict(verdict="keep_srt", final_text=span.srt_text, confidence=0.0,
                             reason="Offline replay: no unambiguous saved decision; source held for review.")
            results.append({**saved, "case_id": span.case_id})
        return results


def deny_network(*args, **kwargs):
    raise RuntimeError("offline replay forbids network access")


def replay(args):
    import dubsync.pipeline as pipeline
    from dubsync.cache import _cache_safe_params

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = Path(manifest.get("repo_root", manifest_path.parent))
    if not base.is_absolute():
        base = manifest_path.parent / base

    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    episode = next(e for e in manifest["episodes"] if e["id"] == args.episode)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.episode):
        raise ValueError("episode ID must contain only letters, numbers, underscores and hyphens")
    transcripts = episode["asr"].get(args.model, [])
    transcript = next(t for t in transcripts if (t["id"] == args.transcript if args.transcript
                                                else t.get("coverage") == "full"))
    normalized = next(c for c in episode["normalized_16k_candidates"] if c["is_full_episode"])
    out = Path(args.out).resolve() / f"{args.episode}.{args.model}.{args.mode}"
    out.mkdir(parents=True, exist_ok=False)
    provider = {}
    if args.providers:
        loaded = yaml.safe_load(Path(args.providers).read_text(encoding="utf-8")) or {}
        # Only local timing/text policies are carried over. Never copy credentials or
        # optional network providers from a production provider configuration.
        provider = {key: _cache_safe_params(copy.deepcopy(loaded[key]))
                    for key in ("timing", "generation", "output", "adjudication") if key in loaded}
        if isinstance(loaded.get("vad"), dict) and loaded["vad"].get("provider", "energy") == "energy":
            provider["vad"] = {k: _cache_safe_params(v) for k, v in loaded["vad"].items()
                               if k in {"provider", "window_ms", "threshold_dbfs", "min_speech_ms",
                                        "min_silence_ms", "padding_ms", "enabled", "min_coverage",
                                        "min_region_ms", "merge_gap_ms", "hysteresis_db", "edge_rise_db",
                                        "boundary_refinement"}}
    fixture = resolve(transcript["fixture_path"])
    audio = resolve(normalized["path"])
    provider["asr"] = {"fixture_path": str(fixture), "model_id":
                       "microsoft/mai-transcribe-2" if args.model == "mai" else "scribe_v2"}
    adapter, decisions_stage = None, None
    if args.mode == "saved":
        candidates = [s for s in episode["stage_dirs"].get(args.model, [])
                      if s.get("has_adjudicate") and s.get("has_align")]
        same = [s for s in candidates if s.get("words_hash") == transcript.get("words_hash")]
        decisions_stage = resolve((same or candidates)[0]["path"])
        adapter = SavedDecisionAdapter(decisions_stage, allow_loose=args.loose)
        provider["llm"] = {"provider": "fixture", "responses": {}}
    provider_path = out / "offline-provider.yaml"
    provider_path.write_text(yaml.safe_dump(provider, sort_keys=False), encoding="utf-8")
    fps = None
    if args.fps == "stage":
        fps = next((float(s["qc"]["fps"]) for s in episode["stage_dirs"].get(args.model, [])
                    if (s.get("qc") or {}).get("fps_source") == "explicit" and s["qc"].get("fps")), None)
    elif args.fps != "auto":
        fps = float(args.fps)
    started = time.perf_counter()
    captured = {}
    verify_stage = pipeline._run_verify_stage

    def capture_verify(**kwargs):
        evidence = kwargs.get("speech_evidence")
        captured.update(
            cue_word_indices={str(key): list(value) for key, value in kwargs["alignment"].cue_word_indices.items()},
            words=[word.model_dump() for word in (evidence.words if evidence is not None else kwargs["words"])],
        )
        return verify_stage(**kwargs)

    with patch.object(socket.socket, "connect", deny_network), patch.object(socket, "create_connection", deny_network):
        with patch.object(pipeline, "llm_adapter_from_config", lambda *a, **k: adapter), \
                patch.object(pipeline, "_run_verify_stage", capture_verify):
            result = pipeline.sync_episode(resolve(episode["source_srt"]["path"]), audio,
                                           out / "output.srt", out / "stages", providers_path=provider_path,
                                           no_llm=args.mode == "nollm", fps=fps)
    (result.episode_workdir / "verify_inputs.json").write_text(json.dumps(captured, ensure_ascii=False), encoding="utf-8")
    report = json.loads((result.episode_workdir / "qc_report.json").read_text(encoding="utf-8"))
    summary = report["summary"]
    kinds = collections.Counter(f"{f['kind']}:{f.get('severity')}" for f in report["flags"])
    row = dict(fps=fps, episode=args.episode, model=args.model, mode=args.mode, transcript=transcript["id"],
               fixture=str(fixture), audio=str(audio), decisions_stage=str(decisions_stage) if decisions_stage else None,
               saved_decisions_matched=len(adapter.matched) if adapter else None,
               saved_decisions_matched_loose=len(adapter.matched_loose) if adapter else None,
               saved_decisions_held=len(adapter.held) if adapter else None,
               elapsed_s=round(time.perf_counter() - started, 2), cues=summary.get("cue_count"),
               flags=len(report["flags"]), error=summary.get("error_count"), warning=summary.get("warning_count"),
               info=summary.get("info_count"), style_violations=summary.get("style_violations"),
               top=kinds.most_common(8), output=str(out / "output.srt"))
    (out / "replay-evidence.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--model", choices=["mai", "scribe_v2"], default="mai")
    parser.add_argument("--transcript")
    parser.add_argument("--mode", choices=["nollm", "saved"], default="nollm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--providers")
    parser.add_argument("--fps", default="stage")
    parser.add_argument("--loose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        print(json.dumps(replay(args), ensure_ascii=False))
    except (ValueError, OSError, KeyError, IndexError, StopIteration) as exc:
        parser.exit(1, f"Replay failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
