"""Run every offline corpus replay against the code in DUBSYNC_SRC and aggregate hold / flag metrics.

  python scripts/bench_corpus.py --manifest <path> --out <fresh-dir> [--jobs 4]

Writes <dir>/summary.json and prints a compact table. Never touches the baseline replay_runs.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONFIDENCE_HOLD_KINDS = {"low_confidence_adjudication", "adjudication_audio_unavailable"}
HOLD_KINDS = CONFIDENCE_HOLD_KINDS | {
    "timing_evidence_held",
    "adjudication_word_mapping_held",
    "missing_audio_timing_held",
    "missing_audio_source_cue_held",
    "unmatched_cue",
    "adjudication_replacement_ownership_held",
}


def run_one(job, out, env, manifest, providers):
    episode, model, transcript, mode = job
    cmd = [sys.executable, "-B", str(HERE / "replay_offline.py"), "--episode", episode, "--model", model,
           "--transcript", transcript, "--mode", mode, "--out", str(out / "runs"),
           "--manifest", str(manifest)]
    if providers:
        cmd.extend(["--providers", str(providers)])
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    if r.returncode != 0:
        return dict(episode=episode, model=model, mode=mode, failed=True, stderr=(r.stderr or r.stdout)[-1500:])
    try:
        last = [line for line in r.stdout.splitlines() if line.startswith("{")][-1]
        row = json.loads(last)
    except (IndexError, json.JSONDecodeError) as exc:
        return dict(episode=episode, model=model, mode=mode, failed=True, stderr=f"Invalid replay evidence: {exc}")
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def is_lyric(lines):
    text = " ".join(lines)
    return "♪" in text or "♫" in text


def is_bracket(lines):
    text = " ".join(lines).strip()
    return text.startswith("[") and text.endswith("]")


def analyse(run_dir: Path):
    stage = glob.glob(str(run_dir / "stages" / "*"))[0]
    rep = json.load(open(stage + "/qc_report.json", encoding="utf-8"))
    ingest = {c["index"]: c for c in json.load(open(stage + "/ingest.json", encoding="utf-8"))["cues"]}
    rebuilt = json.load(open(stage + "/rebuild.json", encoding="utf-8"))["cues"]
    source_timed = [c for c in rebuilt if c["index"] in ingest and ingest[c["index"]]["start_ms"] == c["start_ms"]
                    and ingest[c["index"]]["end_ms"] == c["end_ms"]]
    st_lyric = sum(1 for c in source_timed if is_lyric(ingest[c["index"]]["lines"]))
    st_bracket = sum(1 for c in source_timed if is_bracket(ingest[c["index"]]["lines"]) and not is_lyric(ingest[c["index"]]["lines"]))
    kinds = collections.Counter(f"{f['kind']}:{f.get('severity')}" for f in rep["flags"])
    conf_held = {cid for f in rep["flags"] if f["kind"] in CONFIDENCE_HOLD_KINDS for cid in f.get("cue_ids", [])}
    held = {cid for f in rep["flags"] if f["kind"] in HOLD_KINDS for cid in f.get("cue_ids", [])}
    short = sum(1 for c in rebuilt if c["end_ms"] - c["start_ms"] <= 40)
    fragment = sum(1 for c in rebuilt if len("".join(ch for ch in " ".join(c["lines"]) if ch.isalnum())) <= 1)
    return dict(
        cues=len(rebuilt), source_cues=len(ingest), source_timed=len(source_timed), source_timed_lyric=st_lyric,
        source_timed_bracket=st_bracket, source_timed_dialogue=len(source_timed) - st_lyric - st_bracket,
        confidence_held_cues=len(conf_held), held_cues=len(held), flags=len(rep["flags"]),
        errors=rep["summary"].get("error_count") or 0, warnings=rep["summary"].get("warning_count") or 0,
        flag_errors=rep["summary"].get("flags_by_severity", {}).get("error", 0),
        kinds=dict(kinds), cues_le_40ms=short, fragment_cues=fragment,
    )


def aggregate(out: Path, rows):
    groups = collections.defaultdict(lambda: collections.defaultdict(float))
    kinds = collections.defaultdict(collections.Counter)
    per_run = {}
    for row in rows:
        if row.get("failed"):
            for key in (row["mode"], "ALL"):
                groups[key]["failures"] += 1
            continue
        run_dir = out / "runs" / f"{row['episode']}.{row['model']}.{row['mode']}"
        a = analyse(run_dir)
        per_run[run_dir.name] = {k: v for k, v in a.items() if k != "kinds"}
        keys = [row["mode"], f"{row['mode']}.{row['model']}"]
        if row["episode"].startswith("webjob-"):
            keys.append(f"{row['mode']}.webjobs13")
        for key in keys:
            g = groups[key]
            g["runs"] += 1
            for name in ("cues", "source_cues", "source_timed", "source_timed_lyric", "source_timed_bracket",
                         "source_timed_dialogue", "confidence_held_cues", "held_cues", "flags", "errors", "warnings",
                         "flag_errors", "cues_le_40ms", "fragment_cues"):
                g[name] += a[name]
            kinds[key].update(a["kinds"])
    summary = {}
    for key, g in sorted(groups.items()):
        cues = g.get("cues", 0) or 1
        summary[key] = {**{k: int(v) for k, v in g.items()},
                        "flags_per_100": round(100 * g.get("flags", 0) / cues, 1),
                        "errors_per_100": round(100 * g.get("errors", 0) / cues, 1),
                        "kinds_per_100": {k: round(100 * v / cues, 2) for k, v in kinds[key].most_common()},
                        "kinds": dict(kinds[key].most_common())}
    json.dump(dict(groups=summary, per_run=per_run, failures=[r for r in rows if r.get("failed")]),
              open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for key, s in summary.items():
        print(key, {k: v for k, v in s.items() if k not in ("kinds", "kinds_per_100")})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--providers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if a.jobs < 1:
        ap.error("--jobs must be positive")
    out = Path(a.out).resolve()
    if not a.aggregate_only:
        out.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(a.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = []
    for e in manifest["episodes"]:
        if a.only and a.only not in e["id"]:
            continue
        for c in e.get("offline_replay", {}).get("commands", []):
            for mode in ("nollm", "saved"):
                if mode == "saved" and not c["saved"]:
                    continue
                jobs.append((e["id"], c["model"], c["transcript"], mode))
    rows_path = out / "rows.json"
    if not jobs:
        ap.error("manifest selection contains no replay jobs")
    if a.aggregate_only:
        rows = json.load(open(rows_path, encoding="utf-8"))
    else:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            rows = list(pool.map(lambda j: run_one(j, out, env, manifest_path, a.providers), jobs))
        json.dump(rows, open(rows_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("runs", len(rows), "failed", sum(1 for r in rows if r.get("failed")), "wall", round(time.time() - t0, 1))
        for r in rows:
            if r.get("failed"):
                print("FAIL", r["episode"], r["model"], r["mode"], r["stderr"][-600:].replace("\n", " | "))
    aggregate(out, rows)
    return 1 if any(row.get("failed") for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
