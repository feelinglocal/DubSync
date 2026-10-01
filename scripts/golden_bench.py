"""Golden-reference benchmark for DubSync outputs (read-only; no network).

Usage (from repo root, Git Bash):
    .venv/Scripts/python.exe scripts/golden_bench.py \
        --manifest SUITE.json [--json OUT.json] [--examples N]
    .venv/Scripts/python.exe scripts/golden_bench.py \
        --source S.srt --golden G.srt --output label=PATH [--output label2=PATH2 ...]

What it measures for every (golden, output) pair
  * cue counts; official dubsync.evaluation.evaluate_against_golden metrics (1:1 text-matched cues)
  * boundary-anchored timing: every golden cue whose first (last) token aligns, via a global
    token alignment, to an output token that is also the first (last) token of an output cue
    contributes one start (end) pair.  This survives cue splits/merges and text edits elsewhere
    in the cue.  Errors are output - golden (signed, ms), reported raw and with the output time
    floored to the 30 fps grid (the human references were saved by an editor that floors to
    30 fps frames, so raw comparisons carry a 0..33 ms bias).
  * text: token-level alignment output-vs-golden, attributed with a source-vs-golden alignment
      - captured     : golden differs from source there and the output matches golden
      - missed       : golden differs from source there and the output kept the source words
      - false_edit   : golden kept the source words there and the output changed them
      - both_differ  : golden and output both changed the source, differently
  * word order: transposed tokens (same token deleted+inserted within a short window) and
    cue-ownership shifts (identical token stream, cue boundary placed at a different token).
  * structure: missing golden cues (no token aligned), extra output cues, splits, merges.

Nothing here mutates project files.  Results are deterministic for fixed inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, __import__("os").environ.get("DUBSYNC_SRC") or str(ROOT / "src"))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from dubsync.evaluation import evaluate_against_golden  # noqa: E402
from dubsync.models import Cue  # noqa: E402
from dubsync.srt_io import parse_srt_text  # noqa: E402
from dubsync.tokenize import alphanumeric_signature  # noqa: E402

FPS = 30.0
FRAME_MS = 1000.0 / FPS

# ----------------------------------------------------------------------------- loading

def load_cues(path: Path) -> list[Cue]:
    return parse_srt_text(Path(path).read_text(encoding="utf-8-sig"))


@dataclass
class Stream:
    cues: list[Cue]
    tokens: list[str] = field(default_factory=list)
    tok_cue: list[int] = field(default_factory=list)  # position in cues list
    cue_first: dict[int, int] = field(default_factory=dict)
    cue_last: dict[int, int] = field(default_factory=dict)

    @classmethod
    def build(cls, cues: list[Cue]) -> "Stream":
        s = cls(cues=cues)
        for pos, cue in enumerate(cues):
            sig = alphanumeric_signature(cue.plain_text)
            if not sig:
                continue
            s.cue_first[pos] = len(s.tokens)
            for tok in sig:
                s.tokens.append(tok)
                s.tok_cue.append(pos)
            s.cue_last[pos] = len(s.tokens) - 1
        return s


def floor_frame(ms: int) -> int:
    return int(math.floor(ms * FPS / 1000.0 + 1e-6) * 1000.0 / FPS)


def fmt_ts(ms: int) -> str:
    sign = "-" if ms < 0 else ""
    ms = abs(ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msr = divmod(rem, 1000)
    return f"{sign}{h:02d}:{m:02d}:{s:02d},{msr:03d}"


# ----------------------------------------------------------------------------- loose text

_VARIANTS = {
    "pra": "para", "pro": "para", "ta": "esta", "to": "estou", "tava": "estava", "tavam": "estavam",
    "tao": "estao", "vamo": "vamos", "ce": "voce", "cê": "voce", "sr": "senhor", "sra": "senhora",
    "srta": "senhorita", "dr": "doutor", "dra": "doutora", "num": "nao",
    "um": "1", "uma": "1", "dois": "2", "duas": "2", "tres": "3", "quatro": "4", "cinco": "5",
    "seis": "6", "sete": "7", "oito": "8", "nove": "9", "dez": "10",
}
_INTERJECTIONS = {"hum", "hm", "hmm", "ha", "ah", "oh", "uh", "uhum", "ahn", "eh", "ne", "ai", "ui", "opa",
                  "uau", "ei", "ue", "aha", "hein", "ahh", "ufa", "tsk", "psiu", "oba", "ih", "eee", "eeee"}


def loose_tokens(tokens: list[str]) -> list[str]:
    """Fold orthographic/register variants and drop interjections (Portuguese-oriented)."""
    out = []
    for t in tokens:
        if t in _INTERJECTIONS:
            continue
        out.append(_VARIANTS.get(t, t))
    return out


# ----------------------------------------------------------------------------- alignment

def token_map(a: list[str], b: list[str]):
    """Return (a_to_b, opcodes) where a_to_b[i] = j for equal-aligned tokens else None."""
    ops = Levenshtein.opcodes(a, b)
    a_to_b: list[int | None] = [None] * len(a)
    for op in ops:
        if op.tag == "equal":
            for k in range(op.src_end - op.src_start):
                a_to_b[op.src_start + k] = op.dest_start + k
    return a_to_b, ops


def dist_stats(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    absv = sorted(abs(v) for v in values)
    n = len(values)

    def pct(p: float) -> float:
        if n == 1:
            return absv[0]
        k = (n - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return absv[lo] + (absv[hi] - absv[lo]) * (k - lo)

    return {
        "n": n,
        "median_abs": round(statistics.median(absv), 1),
        "mae": round(sum(absv) / n, 1),
        "p90_abs": round(pct(0.90), 1),
        "p95_abs": round(pct(0.95), 1),
        "max_abs": max(absv),
        "bias_mean_signed": round(sum(values) / n, 1),
        "median_signed": round(statistics.median(values), 1),
        "within_1f": round(sum(1 for v in absv if v <= FRAME_MS + 0.5) / n, 4),
        "within_3f": round(sum(1 for v in absv if v <= 3 * FRAME_MS + 0.5) / n, 4),
        "within_100ms": round(sum(1 for v in absv if v <= 100) / n, 4),
        "within_250ms": round(sum(1 for v in absv if v <= 250) / n, 4),
        "over_500ms": sum(1 for v in absv if v > 500),
        "over_1000ms": sum(1 for v in absv if v > 1000),
    }


# ----------------------------------------------------------------------------- core compare

def compare(source: list[Cue], golden: list[Cue], output: list[Cue], *, timing_valid: bool = True,
            n_examples: int = 40) -> dict:
    S, G, O = Stream.build(source), Stream.build(golden), Stream.build(output)
    g_to_o, go_ops = token_map(G.tokens, O.tokens)
    o_to_g: list[int | None] = [None] * len(O.tokens)
    for gi, oj in enumerate(g_to_o):
        if oj is not None:
            o_to_g[oj] = gi

    result: dict = {
        "cue_count_golden": len(golden),
        "cue_count_output": len(output),
        "tokens_golden": len(G.tokens),
        "tokens_output": len(O.tokens),
    }
    # ---- official metric
    try:
        official = evaluate_against_golden(output, golden, fps=FPS, source=source)
        result["official"] = {k: official[k] for k in (
            "matched_cues", "golden_match_coverage", "start_mae_ms", "end_mae_ms",
            "starts_within_1_frame_ratio", "starts_within_3_frames_ratio",
            "ends_within_1_frame_ratio", "ends_within_3_frames_ratio")}
    except Exception as exc:  # pragma: no cover - diagnostic only
        result["official"] = {"error": repr(exc)}

    # ---- structure
    golden_cue_to_out_cues: dict[int, set[int]] = defaultdict(set)
    out_cue_to_golden_cues: dict[int, set[int]] = defaultdict(set)
    for gi, oj in enumerate(g_to_o):
        if oj is not None:
            golden_cue_to_out_cues[G.tok_cue[gi]].add(O.tok_cue[oj])
            out_cue_to_golden_cues[O.tok_cue[oj]].add(G.tok_cue[gi])
    g_with_tokens = set(G.cue_first)
    o_with_tokens = set(O.cue_first)
    missing = sorted(g for g in g_with_tokens if not golden_cue_to_out_cues.get(g))
    extra = sorted(o for o in o_with_tokens if not out_cue_to_golden_cues.get(o))
    splits = sorted(g for g, outs in golden_cue_to_out_cues.items() if len(outs) > 1)
    merges = sorted(o for o, gs in out_cue_to_golden_cues.items() if len(gs) > 1)
    result["structure"] = {
        "golden_cues_without_any_aligned_output_token": len(missing),
        "output_cues_without_any_aligned_golden_token": len(extra),
        "golden_cues_split_across_output_cues": len(splits),
        "output_cues_spanning_multiple_golden_cues": len(merges),
        "missing_examples": [
            {"g": golden[g].index, "time": f"{fmt_ts(golden[g].start_ms)}-{fmt_ts(golden[g].end_ms)}",
             "text": golden[g].plain_text} for g in missing[:12]],
        "extra_examples": [
            {"o": output[o].index, "time": f"{fmt_ts(output[o].start_ms)}-{fmt_ts(output[o].end_ms)}",
             "text": output[o].plain_text} for o in extra[:12]],
    }

    # ---- boundary-anchored timing
    start_pairs, end_pairs = [], []
    for gpos, gfirst in G.cue_first.items():
        oj = g_to_o[gfirst]
        if oj is None:
            continue
        opos = O.tok_cue[oj]
        if O.cue_first.get(opos) == oj:
            start_pairs.append((gpos, opos))
    for gpos, glast in G.cue_last.items():
        oj = g_to_o[glast]
        if oj is None:
            continue
        opos = O.tok_cue[oj]
        if O.cue_last.get(opos) == oj:
            end_pairs.append((gpos, opos))
    same_seg = sorted(set(start_pairs) & set(end_pairs))

    def deltas(pairs, attr, floor):
        out = []
        for gpos, opos in pairs:
            ov = getattr(output[opos], attr)
            if floor:
                ov = floor_frame(ov)
            out.append(ov - getattr(golden[gpos], attr))
        return out

    timing = {}
    if timing_valid:
        timing = {
            "start_pairs": len(start_pairs),
            "end_pairs": len(end_pairs),
            "same_segmentation_cues": len(same_seg),
            "start_pair_coverage_of_golden": round(len(start_pairs) / max(1, len(G.cue_first)), 4),
            "end_pair_coverage_of_golden": round(len(end_pairs) / max(1, len(G.cue_last)), 4),
            "start_raw": dist_stats(deltas(start_pairs, "start_ms", False)),
            "start_floor30": dist_stats(deltas(start_pairs, "start_ms", True)),
            "end_raw": dist_stats(deltas(end_pairs, "end_ms", False)),
            "end_floor30": dist_stats(deltas(end_pairs, "end_ms", True)),
        }
        # worst errors (floored) for categorisation
        worst = []
        for kind, pairs, attr in (("start", start_pairs, "start_ms"), ("end", end_pairs, "end_ms")):
            for gpos, opos in pairs:
                d = floor_frame(getattr(output[opos], attr)) - getattr(golden[gpos], attr)
                worst.append((abs(d), kind, d, gpos, opos))
        worst.sort(key=lambda r: (-r[0], r[3]))
        timing["worst"] = [
            {
                "edge": kind, "delta_ms": d,
                "g": golden[gpos].index, "g_time": f"{fmt_ts(golden[gpos].start_ms)}-{fmt_ts(golden[gpos].end_ms)}",
                "g_text": golden[gpos].plain_text,
                "o": output[opos].index, "o_time": f"{fmt_ts(output[opos].start_ms)}-{fmt_ts(output[opos].end_ms)}",
                "o_text": output[opos].plain_text,
                "g_start_ms": golden[gpos].start_ms, "g_end_ms": golden[gpos].end_ms,
                "o_start_ms": output[opos].start_ms, "o_end_ms": output[opos].end_ms,
            }
            for _, kind, d, gpos, opos in worst[:n_examples]
        ]
    result["timing"] = timing if timing_valid else {"skipped": "golden timings are not audio-synced"}

    # ---- text attribution
    # Align golden->source in the SAME direction as golden->output so tie-breaking is identical
    # (with output == source every diff is then classified "missed").
    gs_ops = Levenshtein.opcodes(G.tokens, S.tokens)
    g_changed = [True] * len(G.tokens)
    g_src_range: list[tuple[int, int] | None] = [None] * len(G.tokens)
    g_del_points: dict[int, tuple[int, int]] = {}  # golden boundary pos -> source range deleted by human
    for op in gs_ops:
        if op.tag == "equal":
            for k in range(op.src_end - op.src_start):
                g_changed[op.src_start + k] = False
                g_src_range[op.src_start + k] = (op.dest_start + k, op.dest_start + k + 1)
        elif op.tag == "replace":
            for g in range(op.src_start, op.src_end):
                g_src_range[g] = (op.dest_start, op.dest_end)
        elif op.tag == "delete":  # golden tokens absent from source (human/actor insertion)
            for g in range(op.src_start, op.src_end):
                g_src_range[g] = (op.dest_start, op.dest_start)
        elif op.tag == "insert":  # source tokens absent from golden (human deletion)
            g_del_points[op.src_start] = (op.dest_start, op.dest_end)

    def source_tokens_for_golden_span(g0: int, g1: int) -> list[str]:
        lo = hi = None
        for g in range(g0, g1):
            rng = g_src_range[g]
            if rng is None:
                continue
            lo = rng[0] if lo is None else min(lo, rng[0])
            hi = rng[1] if hi is None else max(hi, rng[1])
        for p in range(g0, g1 + 1):
            if p in g_del_points:
                a, b = g_del_points[p]
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
        return S.tokens[lo:hi] if lo is not None else []

    o_is_source = [False] * len(O.tokens)
    for op in Levenshtein.opcodes(S.tokens, O.tokens):
        if op.tag == "equal":
            for k in range(op.dest_start, op.dest_end):
                o_is_source[k] = True

    categories = Counter()
    cat_tokens = Counter()
    cat_examples: dict[str, list] = defaultdict(list)
    for op in go_ops:
        if op.tag == "equal":
            # captured human changes: golden-changed tokens reproduced by output
            n_cap = sum(1 for g in range(op.src_start, op.src_end) if g_changed[g])
            n_cap += sum(1 for p in range(op.src_start + 1, op.src_end) if p in g_del_points)
            if n_cap:
                cat_tokens["captured"] += n_cap
            continue
        g0, g1, o0, o1 = op.src_start, op.src_end, op.dest_start, op.dest_end
        # human side: golden word not in source, or output retains a source word the human removed
        human_changed = any(g_changed[g] for g in range(g0, g1)) or any(o_is_source[o] for o in range(o0, o1))
        # app side: output word not in source, or output dropped a source word the human kept
        app_changed = any(not o_is_source[o] for o in range(o0, o1)) or any(not g_changed[g] for g in range(g0, g1))
        out_toks = O.tokens[o0:o1]
        gold_toks = G.tokens[g0:g1]
        if human_changed and app_changed:
            cat = "both_differ"
        elif human_changed:
            cat = "missed"
        elif app_changed:
            cat = "false_edit"
        else:
            cat = "alignment_noise"
        categories[cat] += 1
        cat_tokens[cat] += max(len(out_toks), len(gold_toks))
        if len(cat_examples[cat]) < 400:
            gpos = G.tok_cue[g0] if g0 < len(G.tokens) else G.tok_cue[-1]
            opos = O.tok_cue[o0] if o0 < len(O.tokens) else O.tok_cue[-1]
            cat_examples[cat].append({
                "op": op.tag, "golden": " ".join(gold_toks), "output": " ".join(out_toks),
                "g": golden[gpos].index, "g_time": fmt_ts(golden[gpos].start_ms),
                "g_text": golden[gpos].plain_text,
                "o": output[opos].index, "o_text": output[opos].plain_text,
            })
    lev = Levenshtein.distance(G.tokens, O.tokens)
    g_loose, o_loose = loose_tokens(G.tokens), loose_tokens(O.tokens)
    lev_loose = Levenshtein.distance(g_loose, o_loose)
    human_changed_tokens = sum(g_changed) + sum(len(v) for v in g_del_points.values())
    result["text"] = {
        "token_levenshtein_vs_golden": lev,
        "wer_vs_golden": round(lev / max(1, len(G.tokens)), 4),
        # orthographic variants folded (pra/para, ta/esta, vamo/vamos, Sr/senhor, pt number words
        # -> digits) and interjection tokens removed: "content" disagreement only
        "loose_levenshtein_vs_golden": lev_loose,
        "loose_wer_vs_golden": round(lev_loose / max(1, len(g_loose)), 4),
        "diff_spans": sum(categories.values()),
        "spans_by_category": dict(categories),
        "tokens_by_category": dict(cat_tokens),
        "human_changed_tokens_total(golden_vs_source)": human_changed_tokens,
        "examples": {k: v[:n_examples] for k, v in cat_examples.items()},
    }
    # cue-level text mismatches for same-segmentation pairs
    cue_mismatch = 0
    for gpos, opos in same_seg:
        if alphanumeric_signature(golden[gpos].plain_text) != alphanumeric_signature(output[opos].plain_text):
            cue_mismatch += 1
    result["text"]["same_segmentation_cue_text_mismatch"] = cue_mismatch

    # ---- word order
    deletes, inserts = [], []  # (golden_position, token)
    for op in go_ops:
        if op.tag in ("delete", "replace"):
            for k, tok in enumerate(G.tokens[op.src_start:op.src_end]):
                deletes.append((op.src_start + k, tok))  # golden token not reproduced in place
        if op.tag in ("insert", "replace"):
            for tok in O.tokens[op.dest_start:op.dest_end]:
                inserts.append((op.src_start, tok))  # output token not in golden at that place
    used = set()
    moved = []
    for gp, tok in deletes:
        for idx, (ip, itok) in enumerate(inserts):
            if idx in used or itok != tok or abs(ip - gp) > 8:
                continue
            used.add(idx)
            moved.append((gp, tok))
            break
    # ownership: compare cue-boundary positions inside equal regions
    own_shift = []
    g_bound = set()
    for pos, last in G.cue_last.items():
        g_bound.add(last)  # boundary after golden token `last`
    o_bound = set(O.cue_last.values())
    # project output boundaries to golden token coordinates
    o_bound_g = set()
    for oj in o_bound:
        gj = o_to_g[oj]
        if gj is not None:
            o_bound_g.add(gj)
    for gb in sorted(g_bound):
        if g_to_o[gb] is None or (gb + 1 < len(G.tokens) and g_to_o[gb + 1] is None):
            continue  # text differs at the boundary; not a pure ownership question
        if gb in o_bound_g:
            continue
        # look for an output boundary 1..4 tokens away inside the same equal run
        for delta in (1, -1, 2, -2, 3, -3, 4, -4):
            cand = gb + delta
            if cand in o_bound_g and cand not in g_bound:
                gpos = G.tok_cue[gb]
                own_shift.append({
                    "shift_tokens": delta,
                    "g": golden[gpos].index, "g_time": f"{fmt_ts(golden[gpos].start_ms)}-{fmt_ts(golden[gpos].end_ms)}",
                    "g_text": golden[gpos].plain_text,
                    "g_next_text": golden[gpos + 1].plain_text if gpos + 1 < len(golden) else "",
                    "o_boundary_after": G.tokens[cand],
                })
                break
    result["order"] = {
        "moved_tokens": len(moved),  # legacy candidate metric, not confirmed word-order errors
        "confirmed_transpositions": len(confirmed_transpositions(G.tokens, O.tokens)),
        "confirmed_transposition_examples": confirmed_transpositions(G.tokens, O.tokens)[:n_examples],
        "moved_examples": [{"golden_pos": gp, "token": t, "g": golden[G.tok_cue[gp]].index,
                            "g_text": golden[G.tok_cue[gp]].plain_text} for gp, t in moved[:n_examples]],
        "ownership_shifts": len(own_shift),
        "ownership_examples": own_shift[:n_examples],
    }
    return result


# ----------------------------------------------------------------------------- human-edit subset

def confirmed_transpositions(reference: list[str], output: list[str]) -> list[dict]:
    """Conservative adjacent block swaps inside an equal, unique-token window.

    The legacy moved-token counter pairs nearby deletes/inserts and can confuse
    repeated dialogue or edit-alignment ties with reordered words. Keep that
    metric for baseline comparisons; count a confirmed swap only with both
    blocks present in reverse order and no added/removed tokens in the window.
    """
    from difflib import SequenceMatcher

    results = []
    operations = SequenceMatcher(a=reference, b=output, autojunk=False).get_opcodes()
    for index in range(len(operations) - 2):
        first, middle, last = operations[index:index + 3]
        if (first[0], middle[0], last[0]) not in (("delete", "equal", "insert"),
                                                                ("insert", "equal", "delete")):
            continue
        a0, a1, b0, b1 = first[1], last[2], first[3], last[4]
        left, right = reference[a0:a1], output[b0:b1]
        if not 2 <= len(left) <= 8 or len(left) != len(right) or len(set(left)) != len(left):
            continue
        if not any(left[split:] + left[:split] == right for split in range(1, len(left))):
            continue
        if any(a0 < item["end"] and a1 > item["start"] for item in results):
            continue
        results.append({"start": a0, "end": a1, "output_start": b0,
                        "reference": left, "output": right})
    return results

def anchored_pairs(golden: list[Cue], output: list[Cue]):
    G, O = Stream.build(golden), Stream.build(output)
    g_to_o, _ = token_map(G.tokens, O.tokens)
    starts, ends = {}, {}
    for gpos, gfirst in G.cue_first.items():
        oj = g_to_o[gfirst]
        if oj is not None and O.cue_first.get(O.tok_cue[oj]) == oj:
            starts[gpos] = O.tok_cue[oj]
    for gpos, glast in G.cue_last.items():
        oj = g_to_o[glast]
        if oj is not None and O.cue_last.get(O.tok_cue[oj]) == oj:
            ends[gpos] = O.tok_cue[oj]
    return starts, ends


def human_edit_subset(golden: list[Cue], base: list[Cue], outputs: dict[str, list[Cue]]) -> dict:
    """Golden was produced by a human editing `base` (times floored to 30 fps).

    edited edges  = golden edge != floor30(base edge) (> 1 ms): the human corrected the app.
    untouched     = golden edge == floor30(base edge): the human accepted the app's time.
    For every other output we report, on each subset, how far it is from golden.
    """
    b_st, b_en = anchored_pairs(golden, base)
    edited = {"start": {}, "end": {}}
    untouched = {"start": set(), "end": set()}
    for edge, pairs, attr in (("start", b_st, "start_ms"), ("end", b_en, "end_ms")):
        for g, b in pairs.items():
            d = floor_frame(getattr(base[b], attr)) - getattr(golden[g], attr)
            if abs(d) > 1:
                edited[edge][g] = d
            else:
                untouched[edge].add(g)
    res = {
        "edited_edges": {e: len(v) for e, v in edited.items()},
        "untouched_edges": {e: len(v) for e, v in untouched.items()},
        "base_error_on_edited": {e: dist_stats(list(v.values())) for e, v in edited.items()},
        "human_edit_direction": {
            e: {"later_than_app": sum(1 for d in v.values() if d < 0),
                "earlier_than_app": sum(1 for d in v.values() if d > 0)}
            for e, v in edited.items()},
        "outputs": {},
    }
    for label, out in outputs.items():
        o_st, o_en = anchored_pairs(golden, out)
        row = {}
        for edge, pairs, attr in (("start", o_st, "start_ms"), ("end", o_en, "end_ms")):
            ed_d, un_d, fixed, same_as_base, worse = [], [], 0, 0, 0
            for g, o in pairs.items():
                d = floor_frame(getattr(out[o], attr)) - getattr(golden[g], attr)
                if g in edited[edge]:
                    ed_d.append(d)
                    bd = edited[edge][g]
                    if abs(d) <= FRAME_MS + 0.5:
                        fixed += 1
                    elif abs(d - bd) <= FRAME_MS + 0.5:
                        same_as_base += 1
                    elif abs(d) > abs(bd):
                        worse += 1
                elif g in untouched[edge]:
                    un_d.append(d)
            row[edge] = {
                "edited_paired": len(ed_d),
                "edited_fixed_within_1f": fixed,
                "edited_reproduces_base_error": same_as_base,
                "edited_worse_than_base": worse,
                "edited_stats": dist_stats(ed_d),
                "untouched_stats": dist_stats(un_d),
            }
        res["outputs"][label] = row
    return res


# ----------------------------------------------------------------------------- reporting

def summarize_row(label: str, r: dict) -> str:
    t = r.get("timing", {})
    tx = r["text"]
    st = r["structure"]
    if "start_floor30" in t:
        s, e = t["start_floor30"], t["end_floor30"]
        timing = (f"{t['start_pairs']:4d}/{t['end_pairs']:4d} | "
                  f"{s['median_abs']:6.1f} {s['mae']:7.1f} {s['p90_abs']:7.1f} {s['p95_abs']:7.1f} "
                  f"{s['within_1f']*100:5.1f}% {s['within_100ms']*100:5.1f}% {s['within_250ms']*100:5.1f}% {s['bias_mean_signed']:+7.1f} | "
                  f"{e['median_abs']:6.1f} {e['mae']:7.1f} {e['p90_abs']:7.1f} {e['p95_abs']:7.1f} "
                  f"{e['within_1f']*100:5.1f}% {e['within_100ms']*100:5.1f}% {e['within_250ms']*100:5.1f}% {e['bias_mean_signed']:+7.1f}")
    else:
        timing = "(timing n/a)"
    sc = tx["spans_by_category"]
    return (f"{label[:40]:40s} {r['cue_count_output']:4d} | {timing} | WER={tx['wer_vs_golden']*100:5.2f}% looseWER={tx['loose_wer_vs_golden']*100:5.2f}% lev={tx['token_levenshtein_vs_golden']:4d} "
            f"miss={sc.get('missed',0):3d} false={sc.get('false_edit',0):3d} both={sc.get('both_differ',0):3d} "
            f"capt_tok={tx['tokens_by_category'].get('captured',0):4d} | moved={r['order']['moved_tokens']:3d} "
            f"own={r['order']['ownership_shifts']:3d} | miss_cues={st['golden_cues_without_any_aligned_output_token']:3d} "
            f"extra={st['output_cues_without_any_aligned_golden_token']:3d} split={st['golden_cues_split_across_output_cues']:3d} "
            f"merge={st['output_cues_spanning_multiple_golden_cues']:3d}")


def run_suite(suite: dict, n_examples: int, include_missing: bool = False) -> dict:
    results = {}
    for ep, spec in suite.items():
        source = load_cues(spec["source"])
        golden = load_cues(spec["golden"])
        timing_valid = spec.get("timing_valid", True)
        outs = dict(spec["outputs"])
        outs.update(spec.get("head_replays", {}))
        ep_res = {"source": str(spec["source"]), "golden": str(spec["golden"]),
                  "cue_count_source": len(source), "cue_count_golden": len(golden),
                  "reference_frame_rate": FPS, "outputs": {}}
        loaded = {}
        for label, path in outs.items():
            if not Path(path).exists():
                if include_missing:
                    ep_res["outputs"][label] = {"missing": str(path)}
                    continue
                raise FileNotFoundError(path)
            output = load_cues(path)
            loaded[label] = output
            r = compare(source, golden, output, timing_valid=timing_valid, n_examples=n_examples)
            r["path"] = str(path)
            ep_res["outputs"][label] = r
        base_label = next((k for k in loaded if k.startswith("base_")), None)
        if timing_valid and base_label:
            ep_res["human_edit_subset"] = human_edit_subset(
                golden, loaded[base_label], {k: v for k, v in loaded.items() if k != base_label})
            ep_res["human_edit_subset"]["base_label"] = base_label
        results[ep] = ep_res
    return results


def print_table(results: dict) -> None:
    hdr = ("label                                    cues | st/en pairs | START(floor30): med  MAE   p90   p95  <=1f  <=100 <=250 bias |"
           " END(floor30): med  MAE   p90   p95  <=1f <=100 <=250 bias | text | order | structure")
    for ep, er in results.items():
        print(f"\n=== {ep}: source {er['cue_count_source']} cues, golden {er['cue_count_golden']} cues")
        print(hdr)
        for label, r in er["outputs"].items():
            if "missing" in r:
                print(f"{label[:40]:40s} MISSING {r['missing']}")
                continue
            print(summarize_row(label, r))
        hs = er.get("human_edit_subset")
        if hs:
            print(f"  human-edit subset (golden derived from {hs['base_label']}): edited edges {hs['edited_edges']}, "
                  f"untouched {hs['untouched_edges']}, direction {hs['human_edit_direction']}")
            for e in ("start", "end"):
                b = hs["base_error_on_edited"][e]
                if b.get("n"):
                    print(f"    base error on edited {e}s: med {b['median_abs']} MAE {b['mae']} p90 {b['p90_abs']} bias {b['bias_mean_signed']}")
            for label, row in hs["outputs"].items():
                parts = []
                for e in ("start", "end"):
                    x = row[e]
                    es, us = x["edited_stats"], x["untouched_stats"]
                    parts.append(
                        f"{e}: edited {x['edited_paired']:3d} fixed {x['edited_fixed_within_1f']:3d} sameAsBase {x['edited_reproduces_base_error']:3d} "
                        f"worse {x['edited_worse_than_base']:3d} (MAE {es.get('mae','-')}) | untouched n={us.get('n',0)} "
                        f"<=1f {us.get('within_1f',0)*100:5.1f}% MAE {us.get('mae','-')} >250ms {sum(1 for _ in ()) if False else ''}")
                print(f"    {label[:38]:38s} " + " || ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="JSON object with a golden_suite mapping; paths relative to repo_root or manifest")
    ap.add_argument("--source")
    ap.add_argument("--golden")
    ap.add_argument("--output", action="append", default=[], help="label=path")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--examples", type=int, default=40)
    args = ap.parse_args()
    if args.golden:
        suite = {"custom": {"source": Path(args.source or args.golden), "golden": Path(args.golden),
                            "outputs": {o.split("=", 1)[0]: Path(o.split("=", 1)[1]) for o in args.output},
                            "timing_valid": not args.no_timing}}
    elif args.manifest:
        manifest_path = Path(args.manifest).resolve()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = Path(data.get("repo_root", manifest_path.parent))
        if not base.is_absolute():
            base = manifest_path.parent / base
        suite = data["golden_suite"]
        for spec in suite.values():
            for key in ("source", "golden"):
                path = Path(spec[key])
                spec[key] = path if path.is_absolute() else base / path
            for key in ("outputs", "head_replays"):
                spec[key] = {label: Path(path) if Path(path).is_absolute() else base / path
                             for label, path in spec.get(key, {}).items()}
    else:
        ap.error("provide --manifest or --golden with at least one --output")
    if not suite or any(not (spec.get("outputs") or spec.get("head_replays")) for spec in suite.values()):
        ap.error("each suite episode needs at least one output")
    results = run_suite(suite, args.examples)
    print_table(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
