"""Timing metrics for cached pipeline outputs (read-only, standard library only).

    python scripts/timing_vs_audio.py --manifest suite.json [--json metrics.json]

Per run: final start-onset / end-offset distributions (historical benchmark definition:
10 ms envelope, -45 dBFS bursts, isolated edges), first-word-cut count, overlap pairs in the exported SRT,
cues under the minimum duration with free room, and counts of timing-related QC flags.
The manifest contains timing_runs: a list of directories from replay_offline.py
(replay-evidence.json, output.srt, stages/<episode>) or the historical timing
harness (replay-info.json, synced.srt, work/<episode>). Paths use repo_root when supplied,
otherwise the manifest directory. A directory root can be supplied instead.
Default measurements use raw ASR probes for comparison with historical baselines;
--word-evidence effective explicitly uses the repaired words captured at verification.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import math
import statistics
import wave
from array import array
from bisect import bisect_left, bisect_right

TIMING_FLAG_KINDS = [
    "timing_refined", "asr_word_clamped", "asr_word_edges_repaired", "cue_without_speech_activity",
    "cue_with_excessive_trailing_silence", "cue_on_silence", "timing_outlier_trimmed", "min_duration_unattainable",
    "timing_evidence_held", "timing_refinement_held", "overlap_stacked", "output_overlap_unresolved",
    "output_overlap_preserved", "output_overlap_resolved", "adlib_removed_without_speech_activity",
    "dropped_line_candidate", "unmatched_cue", "output_order_inversion",
]
STYLE_KINDS = ["min_duration", "frame_grid", "overlap"]
TS = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s+-->\s+(\d\d):(\d\d):(\d\d),(\d\d\d)")
WORD = re.compile(r"(\S+)\(([\d.]+)-([\d.]+)\)")


def percentile(values, pct):
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def pcts(values):
    a = [value * 1000 for value in values]
    if not a:
        return {"n": 0}
    return {"n": len(a), **{f"p{pct:02d}": round(percentile(a, pct)) for pct in (5, 25, 50, 75, 95)},
            "mean_abs": round(statistics.mean(abs(v) for v in a)),
            "within40": round(sum(abs(v) <= 40 for v in a) / len(a) * 100, 1),
            "within80": round(sum(abs(v) <= 80 for v in a) / len(a) * 100, 1)}


def envelope(path):
    """Stream 16-bit PCM; retain only the 20 ms RMS envelope at 10 ms hops.

    Matches the original benchmark's first-channel / 32767 scaling. Audio samples
    are bounded to two hops; the envelope uses about 100 floats per second.
    """
    db = []
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2:
            raise ValueError("timing benchmark requires 16-bit PCM WAV")
        rate, channels = stream.getframerate(), stream.getnchannels()
        hop, window = int(rate * .01), int(rate * .02)
        if hop < 1 or window != 2 * hop:
            raise ValueError("sample rate must support exact 10 ms hops")
        previous = None
        while raw := stream.readframes(hop):
            samples = array("h")
            samples.frombytes(raw)
            if sys.byteorder != "little":
                samples.byteswap()
            samples = samples[::channels]
            if len(samples) < hop:
                break
            squares = sum((sample / 32767.0) ** 2 for sample in samples)
            if previous is not None:
                db.append(10 * math.log10(max((previous + squares) / window, 1e-12)))
            previous = squares
    return db


def runs(active):
    start = 0
    for index in range(1, len(active) + 1):
        if index == len(active) or active[index] != active[start]:
            yield start, index, active[start]
            start = index


def bursts(db, thr=-45.0, sil=8, act=5):
    active = [value >= thr for value in db]
    for start, end, value in list(runs(active)):
        if not value and end - start < sil and start > 0 and end < len(active):
            active[start:end] = [True] * (end - start)
    for start, end, value in list(runs(active)):
        if value and end - start < act:
            active[start:end] = [False] * (end - start)
    kept = [(start * .01, end * .01) for start, end, value in runs(active) if value]
    return [start for start, _ in kept], [end for _, end in kept]


def burst_of(time, starts, ends):
    index = bisect_right(starts, time) - 1
    return index if index >= 0 and ends[index] >= time else None


def fmt(d):
    if not d.get("n"):
        return "n=0"
    return (f"n={d['n']} p05={d['p05']:+d} p25={d['p25']:+d} p50={d['p50']:+d} p75={d['p75']:+d} p95={d['p95']:+d} "
            f"mean|x|={d['mean_abs']} |<=40|={d['within40']}% |<=80|={d['within80']}%")


def srt_cues(path):
    cues = []
    blocks = Path(path).read_text(encoding="utf-8-sig").split("\n\n")
    for b in blocks:
        m = TS.search(b)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        s = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
        e = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
        text = b[m.end():].strip()
        cues.append((s, e, text))
    return cues


def main(root: Path, out_json: Path | None, manifest: Path | None = None, word_evidence: str = "raw"):
    result = {}
    if manifest:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        base = Path(data.get("repo_root", manifest.resolve().parent))
        if not base.is_absolute():
            base = manifest.resolve().parent / base
        run_dirs = [Path(value) if Path(value).is_absolute() else base / value for value in data["timing_runs"]]
    else:
        run_dirs = sorted(p for p in root.iterdir() if any((p / name).exists()
                          for name in ("replay-info.json", "replay-evidence.json")))
    for run_dir in run_dirs:
        metadata = run_dir / "replay-info.json"
        promoted = not metadata.exists()
        if promoted:
            metadata = run_dir / "replay-evidence.json"
        info = json.loads(metadata.read_text(encoding="utf-8"))
        stages = [path for path in (run_dir / ("stages" if promoted else "work")).iterdir()
                  if path.is_dir() and (path / "qc_report.json").exists()]
        if len(stages) != 1:
            raise ValueError(f"expected exactly one complete episode in {run_dir}")
        work = stages[0]
        db = envelope(info["audio"])
        bs45, be45 = bursts(db, -45.0)
        bs50, be50 = bursts(db, -50.0)
        vin = json.loads((work / "verify_inputs.json").read_text(encoding="utf-8"))
        words = (vin["words"] if word_evidence == "effective" else
                 json.loads((work / "asr.json").read_text(encoding="utf-8"))["words"])
        cwi = {int(k): v for k, v in vin["cue_word_indices"].items()}
        align_cwi = {int(k): v for k, v in json.loads((work / "align.json").read_text(encoding="utf-8"))["cue_word_indices"].items()}
        qc = json.loads((work / "qc_report.json").read_text(encoding="utf-8"))
        reb = json.loads((work / "rebuild.json").read_text(encoding="utf-8"))["cues"]
        source = {c["index"]: c for c in json.loads((work / "ingest.json").read_text(encoding="utf-8"))["cues"]}
        profile = json.loads((work / "style_profile.json").read_text(encoding="utf-8"))
        min_dur_ms = profile["min_cue_dur"] * 1000
        word_ends = sorted(w["end"] for w in words)
        word_starts = sorted(w["start"] for w in words)

        def prev_end(t):
            k = bisect_right(word_ends, t + 1e-6) - 1
            return word_ends[k] if k >= 0 else -1.0

        def next_start_after(t, exclude_start=None):
            k = bisect_left(word_starts, t - 1e-6)
            while k < len(word_starts) and exclude_start is not None and abs(word_starts[k] - exclude_start) < 1e-9:
                k += 1
            return word_starts[k] if k < len(word_starts) else 1e9

        row = {"cues": len(reb), "flags": info["flags"],
               "errors": info["error" if promoted else "errors"],
               "warnings": info["warning" if promoted else "warnings"],
               "style": info["style_violations"]}
        if word_evidence != "raw":
            row["word_evidence"] = word_evidence
        # --- A. final start/end vs bursts (refine_effect definition), for both ownership maps
        for label, mapping in (("align", align_cwi), ("final", cwi)):
            fin_s, fin_e = [], []
            for cue in reb:
                idx = [i for i in mapping.get(cue["index"], []) if 0 <= i < len(words)]
                if not idx:
                    continue
                src = source.get(cue["index"])
                # an edge still at its source time is a hold, not an acoustic boundary
                src_start = bool(src) and src["start_ms"] == cue["start_ms"] and label == "final"
                src_end = bool(src) and src["end_ms"] == cue["end_ms"] and label == "final"
                first = min((words[i] for i in idx), key=lambda w: w["start"])
                last = max((words[i] for i in idx), key=lambda w: w["end"])
                k = burst_of(first["start"] + 0.03, bs45, be45)
                if k is not None and bs45[k] > prev_end(first["start"]) and not src_start:
                    fin_s.append(cue["start_ms"] / 1000 - bs45[k])
                probe = max(last["start"], last["end"] - 0.05)
                k = burst_of(probe, bs45, be45)
                if k is not None and not src_end:
                    nxt = next_start_after(last["end"], exclude_start=None)
                    if nxt >= be45[k]:
                        fin_e.append(cue["end_ms"] / 1000 - be45[k])
            row[f"start_minus_onset[{label}]"] = pcts(fin_s)
            row[f"end_minus_offset[{label}]"] = pcts(fin_e)
        # --- B. first word cut: cue starts > 80 ms after the onset of the burst in which its first word is spoken
        cut = []
        for cue in reb:
            idx = [i for i in cwi.get(cue["index"], []) if 0 <= i < len(words)]
            if not idx:
                continue
            src = source.get(cue["index"])
            if src and src["start_ms"] == cue["start_ms"]:
                continue
            first = min((words[i] for i in idx), key=lambda w: w["start"])
            if first["end"] - first["start"] < 0.05:
                continue
            k = burst_of(first["end"] - 0.03, bs50, be50)
            if k is None:
                continue
            onset = bs50[k]
            if first["start"] > onset + 0.02:
                continue  # word begins inside a running burst: onset belongs to earlier speech
            late = cue["start_ms"] / 1000 - onset
            if late > 0.08:
                cut.append((cue["index"], round(late * 1000), first["text"], first["start"], first["end"], cue["start_ms"] / 1000))
        row["first_word_cut"] = {"count": len(cut), "late_ms": sorted(c[1] for c in cut), "examples": cut[:25]}
        # original flag-based definition (first_word_cut.py)
        flag_cut = 0
        for f in qc["flags"]:
            if f["kind"] != "timing_outlier_trimmed":
                continue
            old = WORD.findall(f.get("old_text") or "")
            new = WORD.findall(f.get("new_text") or "")
            if old and new and old[0] != new[0]:
                flag_cut += 1
        row["first_word_dropped_by_outlier_trim_flags"] = flag_cut
        # --- C. overlaps in the exported SRT
        out = sorted(srt_cues(run_dir / ("output.srt" if promoted else "synced.srt")))
        pairs, pairs_dialogue = 0, 0
        latest = None
        for c in out:
            if latest is not None and c[0] < latest[1]:
                pairs += 1
                bracket = lambda t: t.strip().startswith("[") and t.strip().endswith("]")
                if not bracket(c[2]) and not bracket(latest[2]):
                    pairs_dialogue += 1
            if latest is None or c[1] > latest[1]:
                latest = c
        row["overlap_pairs"] = pairs
        row["overlap_pairs_dialogue_only"] = pairs_dialogue
        # --- D. cues under min duration with free room
        ordered = sorted(reb, key=lambda c: (c["start_ms"], c["end_ms"]))
        short, short_room = 0, 0
        for i, c in enumerate(ordered):
            dur = c["end_ms"] - c["start_ms"]
            if dur >= min_dur_ms - 1:
                continue
            short += 1
            nxt = next((n for n in ordered[i + 1:] if n["start_ms"] >= c["end_ms"]), None)
            if nxt is None or nxt["start_ms"] - c["start_ms"] >= min_dur_ms:
                short_room += 1
        row["cues_under_min_duration"] = short
        row["cues_under_min_duration_with_free_room"] = short_room
        # --- E. flags
        kinds = collections.Counter(f["kind"] for f in qc["flags"])
        styles = collections.Counter(f["kind"] for f in qc["style_issues"])
        row["timing_flags"] = {k: kinds.get(k, 0) for k in TIMING_FLAG_KINDS if kinds.get(k, 0)}
        row["style_issues"] = {k: styles.get(k, 0) for k in STYLE_KINDS if styles.get(k, 0)}
        result[run_dir.name] = row
        print(f"== {run_dir.name}: cues={row['cues']} flags={row['flags']} style={row['style']} err={row['errors']} warn={row['warnings']}")
        for label in ("align", "final"):
            print(f"   start-onset[{label}]: {fmt(row[f'start_minus_onset[{label}]'])}")
            print(f"   end-offset [{label}]: {fmt(row[f'end_minus_offset[{label}]'])}")
        print(f"   first_word_cut={len(cut)} late_ms={row['first_word_cut']['late_ms']} (flag-based: {flag_cut})")
        print(f"   overlap_pairs={pairs} (dialogue-only {pairs_dialogue})  under_min_dur={short} with_free_room={short_room}")
        print(f"   timing flags: {row['timing_flags']}")
        print(f"   style: {row['style_issues']}")
    if not result:
        raise ValueError("no timing runs found; expected replay-info.json in each run directory")
    if out_json:
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--manifest", help="JSON containing timing_runs paths relative to repo_root or manifest")
    ap.add_argument("--json")
    ap.add_argument("--word-evidence", choices=["raw", "effective"], default="raw")
    a = ap.parse_args()
    main(Path(a.root), Path(a.json) if a.json else None, Path(a.manifest) if a.manifest else None,
         word_evidence=a.word_evidence)
