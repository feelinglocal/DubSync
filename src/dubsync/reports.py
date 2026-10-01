from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path

from .cache import write_json_atomic, write_text_atomic
from .models import Cue, CueScore, QCFlag, StyleIssue
from .qc_review import build_review, strip_route_tags
from .srt_io import format_timestamp, write_srt


def write_qc_report(
    report_json_path: Path,
    report_html_path: Path,
    cues: list[Cue],
    flags: list[QCFlag],
    style_issues: list[StyleIssue],
    cue_scores: list[CueScore] | None = None,
    summary_metadata: Mapping[str, object] | None = None,
    *,
    source_cues: list[Cue] | None = None,
) -> dict[str, object]:
    """Write the QC report for ``cues``, the delivered list in SRT order.

    ``flags`` and ``style_issues`` stay verbatim (raw findings for operators and
    tests); ``review``/``changes``/``notes``/``diagnostics`` are the customer view
    from ``qc_review``, numbered like the delivered (renumbered) SRT.
    """

    ordered_flags = _sorted_flags(flags, cues)
    ordered_issues = _sorted_style_issues(style_issues)
    all_findings = [*ordered_flags, *ordered_issues]
    review = build_review(
        ordered_flags,
        ordered_issues,
        cues,
        source_cues=source_cues,
        summary_metadata=summary_metadata,
    )
    summary: dict[str, object] = {
        **dict(summary_metadata or {}),
        "cue_count": len(cues),
        "flags": len(ordered_flags),
        "style_violations": len(ordered_issues),
        "flags_by_severity": _severity_counts(ordered_flags),
        "style_issues_by_severity": _severity_counts(ordered_issues),
        "error_count": sum(1 for item in all_findings if item.severity == "error"),
        "warning_count": sum(1 for item in all_findings if item.severity == "warning"),
        "info_count": sum(1 for item in all_findings if item.severity == "info"),
        "verdict": review.verdict,
        **review.counts,
    }
    payload: dict[str, object] = {
        "summary": summary,
        "review": [item.model_dump() for item in review.review],
        "changes": [item.model_dump() for item in review.changes],
        "notes": [item.model_dump() for item in review.notes],
        "diagnostics": [item.model_dump() for item in review.diagnostics],
        "cue_scores": [score.model_dump() for score in cue_scores or []],
        "flags": [flag.model_dump() for flag in ordered_flags],
        "style_issues": [issue.model_dump() for issue in ordered_issues],
    }
    write_json_atomic(report_json_path, payload)
    write_text_atomic(report_html_path, _render_html(payload))
    return payload


def write_changes_diff(path: Path, flags: list[QCFlag]) -> None:
    """Dump every flag that carries old/new text as SRT review markers.

    Kept for callers that want a raw flag view; the pipeline's customer change
    log is ``write_change_log`` (text changes only, delivered numbering).
    """

    cues: list[Cue] = []
    for flag in flags:
        if flag.old_text is None and flag.new_text is None:
            continue
        cue_ids = ", ".join(str(cue_id) for cue_id in flag.cue_ids) or "ad-lib"
        lines = [f"# {flag.kind} cue={cue_ids}"]
        if flag.old_text is not None:
            lines.extend(f"- {line}" for line in flag.old_text.splitlines())
        if flag.new_text is not None:
            lines.extend(f"+ {line}" for line in flag.new_text.splitlines())
        start_ms = _flag_seconds_to_ms(flag.start)
        end_ms = _flag_seconds_to_ms(flag.end)
        if end_ms <= start_ms:
            # These are review markers, not dialogue. A point finding needs a
            # representable interval while its actual evidence stays visible.
            lines.insert(
                1,
                "# 1 ms diagnostic marker; original timing (seconds): "
                f"{flag.start} --> {flag.end}",
            )
            end_ms = start_ms + 1
        cues.append(
            Cue(
                index=len(cues) + 1,
                start_ms=start_ms,
                end_ms=end_ms,
                lines=lines,
            )
        )
    write_text_atomic(path, write_srt(cues, renumber=True) if cues else "")


def write_change_log(path: Path, changes: Sequence[Mapping[str, object]]) -> None:
    """Write the text change log (``changes.diff.srt``) from the report's ``changes`` list.

    One block per wording change, added or removed line, in playback order and
    numbered like the delivered SRT; timing-only changes stay in the QC report.
    Each block is a valid SRT cue whose text is the diff, so it can be loaded
    next to the delivered file in a subtitle editor.
    """

    blocks: list[Cue] = []
    for change in changes:
        if change.get("change") not in ("edited", "added", "removed"):
            continue
        lines = [_change_header(change)]
        reason = change.get("reason")
        if isinstance(reason, str) and reason.strip():
            lines.append(f"# {reason.strip()}")
        old_text, new_text = change.get("old_text"), change.get("new_text")
        if isinstance(old_text, str) and old_text:
            lines.extend(f"- {line}" for line in old_text.splitlines())
        if isinstance(new_text, str) and new_text:
            lines.extend(f"+ {line}" for line in new_text.splitlines())
        start_ms = _flag_seconds_to_ms(_float_or_none(change.get("start")))
        end_ms = _flag_seconds_to_ms(_float_or_none(change.get("end")))
        if end_ms <= start_ms:
            lines.insert(1, f"# 1 ms marker; original timing (seconds): {change.get('start')} --> {change.get('end')}")
            end_ms = start_ms + 1
        blocks.append(Cue(index=len(blocks) + 1, start_ms=start_ms, end_ms=end_ms, lines=lines))
    blocks.sort(key=lambda block: (block.start_ms, block.index))
    write_text_atomic(path, write_srt(blocks, renumber=True) if blocks else "")


def _change_header(change: Mapping[str, object]) -> str:
    kind = change.get("change")
    cue_id = change.get("cue_id")
    internal = f" (cue {cue_id})" if cue_id is not None else ""
    number = change.get("srt_number")
    if isinstance(number, int):
        return f"# SRT #{number} {kind}{internal}"
    after = change.get("after_srt_number")
    where = f" after SRT #{after}" if isinstance(after, int) else " before SRT #1"
    return f"# {kind}{where}{internal}"


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


_VERDICT_BANNERS = {
    "clean": "Ready: nothing needs review",
    "check": "Check",
    "attention": "Attention needed",
}

_HTML_STYLE = (
    ":root{color-scheme:light dark;--ink:#091717;--muted:#586767;--line:#d7dede;--surface:#f6f8f8;"
    "--canvas:#fff;--error:#b42318;--error-bg:#fff4f2;--warn:#7a5a00;--warn-bg:#fffde3;--ok:#087a55;"
    "--ok-bg:#eefaf4;--old:#9a2b1f;--new:#0b6b3a}"
    "@media (prefers-color-scheme:dark){:root{--ink:#edf4f4;--muted:#a6b5b5;--line:#2b383e;--surface:#151f24;"
    "--canvas:#0d1418;--error:#ff9b90;--error-bg:#321d1d;--warn:#f0d264;--warn-bg:#292710;--ok:#5bd6aa;"
    "--ok-bg:#10261e;--old:#ff9b90;--new:#5bd6aa}}"
    "body{margin:0;padding:24px 16px 48px;font:14px/1.45 'Segoe UI',system-ui,Arial,sans-serif;"
    "color:var(--ink);background:var(--canvas)}"
    "main{max-width:1180px;margin:0 auto}h1{font-size:22px;margin:0 0 8px}"
    "h2{font-size:17px;margin:28px 0 8px}.meta{color:var(--muted);margin:4px 0}"
    ".banner{border-radius:8px;padding:12px 14px;margin:12px 0;font-weight:600}"
    ".banner.clean{background:var(--ok-bg);color:var(--ok)}.banner.check{background:var(--warn-bg);color:var(--warn)}"
    ".banner.attention{background:var(--error-bg);color:var(--error)}"
    ".scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}"
    "td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}"
    "th{background:var(--surface);font-weight:600}td.num,td.time{white-space:nowrap;font-variant-numeric:tabular-nums}"
    ".chip{display:inline-block;border-radius:999px;padding:1px 8px;font-size:12px;font-weight:600}"
    ".chip.error{background:var(--error-bg);color:var(--error)}.chip.warning{background:var(--warn-bg);color:var(--warn)}"
    ".old{color:var(--old)}.new{color:var(--new)}.muted{color:var(--muted)}"
    "details{margin:12px 0}summary{cursor:pointer;font-weight:600}ul{padding-left:20px}"
)


def _render_html(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    review = _dict_items(payload.get("review"))
    changes = _dict_items(payload.get("changes"))
    notes = _dict_items(payload.get("notes"))
    diagnostics = _dict_items(payload.get("diagnostics"))
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>DubSync QC report</title><style>{_HTML_STYLE}</style></head><body><main>",
        "<h1>DubSync QC report</h1>",
        _render_banner(summary),
        _render_meta(summary),
        _render_review(review),
        _render_changes(changes),
        _render_notes(notes),
        _render_diagnostics(payload, diagnostics),
        "</main></body></html>",
    ]
    return "".join(parts)


def _render_banner(summary: Mapping[str, object]) -> str:
    verdict = summary.get("verdict")
    if verdict not in _VERDICT_BANNERS:
        return ""
    errors = _count(summary.get("review_error_count"))
    warnings = _count(summary.get("review_warning_count"))
    cues = _count(summary.get("review_cue_count"))
    if verdict == "clean":
        text = _VERDICT_BANNERS["clean"]
    else:
        details = [
            _plural(errors, "item") + (" needs" if errors == 1 else " need") + " fixing" if errors else "",
            f"{_plural(warnings, 'item')} to check" if warnings else "",
        ]
        text = f"{_VERDICT_BANNERS[str(verdict)]}: {' · '.join(item for item in details if item)} ({_plural(cues, 'cue')})"
    return f"<p class=\"banner {html.escape(str(verdict))}\">{html.escape(text)}</p>"


def _render_meta(summary: Mapping[str, object]) -> str:
    facts = [f"{_plural(_count(summary.get('cue_count')), 'cue')} delivered"]
    if "change_count" in summary:
        facts.append(
            f"{_plural(_count(summary.get('change_count')), 'change')} logged "
            f"({_count(summary.get('text_change_count'))} wording, {_count(summary.get('timing_change_count'))} timing)"
        )
    if "note_count" in summary:
        facts.append(_plural(_count(summary.get("note_count")), "note"))
    fps = summary.get("fps")
    if isinstance(fps, int | float) and not isinstance(fps, bool):
        source = summary.get("fps_source")
        facts.append(f"{fps:g} fps" + (f" ({html.escape(str(source))})" if source else ""))
    facts.append(f"{_count(summary.get('flags')) + _count(summary.get('style_violations'))} raw findings")
    return f"<p class=\"meta\">{html.escape(' · '.join(facts))}</p>"


def _render_review(items: list[dict[str, object]]) -> str:
    if not items:
        return "<h2>Needs review (0)</h2><p class=\"muted\">Nothing needs a human check.</p>"
    rows = []
    for item in items:
        severity = str(item.get("severity", "warning"))
        diff = ""
        if isinstance(item.get("old_text"), str) and isinstance(item.get("new_text"), str):
            diff = (
                f"<div><span class=\"old\">{_format_multiline(item['old_text'])}</span> &rarr; "
                f"<span class=\"new\">{_format_multiline(item['new_text']) or '(removed)'}</span></div>"
            )
        rows.append(
            "<tr>"
            f"<td class=\"num\">{html.escape(_location(item))}</td>"
            f"<td class=\"time\">{html.escape(str(item.get('timecode') or ''))}</td>"
            f"<td><span class=\"chip {html.escape(severity)}\">{'fix' if severity == 'error' else 'check'}</span></td>"
            f"<td><strong>{html.escape(str(item.get('title', '')))}</strong><br>"
            f"<span class=\"muted\">{html.escape(str(item.get('detail', '')))}</span>{diff}</td>"
            f"<td>{_format_multiline(item.get('text'))}</td>"
            f"<td>{html.escape(str(item.get('action', '')))}</td>"
            "</tr>"
        )
    return (
        f"<h2>Needs review ({len(items)})</h2><div class=\"scroll\"><table>"
        "<tr><th>SRT #</th><th>Time</th><th></th><th>Issue</th><th>Subtitle</th><th>What to do</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def _render_changes(items: list[dict[str, object]]) -> str:
    if not items:
        return "<h2>Changes (0)</h2><p class=\"muted\">The subtitles keep your wording; no timing changes were logged.</p>"
    rows = []
    for item in items:
        if item.get("change") == "timing":
            before, after = item.get("old_timing"), item.get("new_timing")
        else:
            before, after = item.get("old_text"), item.get("new_text")
        rows.append(
            "<tr>"
            f"<td class=\"num\">{html.escape(_location(item))}</td>"
            f"<td class=\"time\">{html.escape(str(item.get('timecode') or ''))}</td>"
            f"<td>{html.escape(str(item.get('title', '')))}</td>"
            f"<td class=\"old\">{_format_multiline(before)}</td>"
            f"<td class=\"new\">{_format_multiline(after)}</td>"
            f"<td class=\"muted\">{html.escape(str(item.get('reason') or ''))}</td>"
            "</tr>"
        )
    return (
        f"<h2>Changes ({len(items)})</h2><div class=\"scroll\"><table>"
        "<tr><th>SRT #</th><th>Time</th><th>Change</th><th>Before</th><th>After</th><th>Reason</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def _render_notes(items: list[dict[str, object]]) -> str:
    if not items:
        return "<h2>Notes (0)</h2>"
    entries = "".join(
        f"<li><strong>{html.escape(str(item.get('title', '')))}</strong> &mdash; "
        f"{html.escape(str(item.get('detail', '')))}</li>"
        for item in items
    )
    return f"<h2>Notes ({len(items)})</h2><ul>{entries}</ul>"


def _render_diagnostics(payload: Mapping[str, object], diagnostics: list[dict[str, object]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('kind', '')))}</td>"
        f"<td class=\"num\">{html.escape(str(item.get('count', '')))}</td>"
        f"<td>{html.escape(str(item.get('title', '')))}</td>"
        f"<td class=\"muted\">{html.escape(str(item.get('message', '')))}</td>"
        "</tr>"
        for item in diagnostics
    )
    total = sum(_count(item.get("count")) for item in diagnostics)
    return (
        f"<details><summary>Diagnostics (technical, {total} findings)</summary>"
        "<div class=\"scroll\"><table><tr><th>Kind</th><th>Count</th><th>Meaning</th><th>Example</th></tr>"
        + rows
        + "</table></div>"
        + _render_cue_scores(payload.get("cue_scores"))
        + _render_raw_findings(payload)
        + "</details>"
    )


def _render_cue_scores(cue_scores: object) -> str:
    # Only cues with real evidence: MAI words carry no confidence and a
    # constant ASR confidence says nothing about a cue.
    scored = [
        item for item in _dict_items(cue_scores)
        if item.get("source") != "unscored" and isinstance(item.get("score"), int | float)
    ]
    if not scored:
        return "<p class=\"muted\">Cue scores: no per-cue confidence evidence for this transcription.</p>"
    rows = "".join(
        "<tr>"
        f"<td class=\"num\">{html.escape(str(item.get('cue_id', '')))}</td>"
        f"<td class=\"num\">{html.escape(str(item.get('cps', '')))}</td>"
        f"<td class=\"num\">{html.escape(str(item.get('score', '')))}</td>"
        f"<td>{html.escape(str(item.get('source', '')))}</td>"
        "</tr>"
        for item in scored
    )
    return (
        f"<details><summary>Cue scores ({len(scored)})</summary><div class=\"scroll\"><table>"
        "<tr><th>Cue id</th><th>CPS</th><th>Score</th><th>Source</th></tr>" + rows + "</table></div></details>"
    )


def _render_raw_findings(payload: Mapping[str, object]) -> str:
    flags = _dict_items(payload.get("flags"))
    issues = _dict_items(payload.get("style_issues"))
    flag_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('severity', '')))}</td>"
        f"<td>{html.escape(str(item.get('kind', '')))}</td>"
        f"<td>{html.escape(', '.join(str(cue_id) for cue_id in item.get('cue_ids') or []))}</td>"
        f"<td>{_format_seconds(item.get('start'))}</td>"
        f"<td>{_format_seconds(item.get('end'))}</td>"
        f"<td>{html.escape(strip_route_tags(str(item.get('message', ''))))}</td>"
        f"<td>{html.escape(str(item.get('confidence', '')) if item.get('confidence') is not None else '')}</td>"
        f"<td>{_format_multiline(item.get('old_text'))}</td>"
        f"<td>{_format_multiline(item.get('new_text'))}</td>"
        "</tr>"
        for item in flags
    )
    issue_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('severity', '')))}</td>"
        f"<td>{html.escape(str(item.get('kind', '')))}</td>"
        f"<td>{html.escape(str(item.get('cue_id', '')))}</td>"
        f"<td>{html.escape(str(item.get('message', '')))}</td>"
        "</tr>"
        for item in issues
    )
    return (
        f"<details><summary>All raw findings ({len(flags)} flags, {len(issues)} style issues; internal cue ids)"
        "</summary><div class=\"scroll\"><table><tr><th>Severity</th><th>Kind</th><th>Cue ids</th><th>Start</th>"
        "<th>End</th><th>Message</th><th>Value</th><th>Old text</th><th>New text</th></tr>"
        + flag_rows
        + "</table><table><tr><th>Severity</th><th>Kind</th><th>Cue id</th><th>Message</th></tr>"
        + issue_rows
        + "</table></div></details>"
    )


def _location(item: Mapping[str, object]) -> str:
    label = item.get("srt_label")
    if isinstance(label, str) and label:
        return label
    after = item.get("after_srt_number")
    if isinstance(after, int):
        return f"after #{after}"
    return "—"


def _dict_items(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _plural(count: int, label: str) -> str:
    return f"{count} {label}{'' if count == 1 else 's'}"


_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}


def _sorted_flags(flags: list[QCFlag], cues: list[Cue]) -> list[QCFlag]:
    cue_starts = {cue.index: cue.start_ms / 1000.0 for cue in cues}
    return [
        flag
        for _original_index, flag in sorted(
            enumerate(flags),
            key=lambda item: _flag_sort_key(item[0], item[1], cue_starts),
        )
    ]


def _flag_sort_key(
    original_index: int,
    flag: QCFlag,
    cue_starts: dict[int, float],
) -> tuple[int, float, int, int]:
    cue_id = min(flag.cue_ids) if flag.cue_ids else 1_000_000_000
    cue_position = min(
        (cue_starts[flag_cue_id] for flag_cue_id in flag.cue_ids if flag_cue_id in cue_starts),
        default=float("inf"),
    )
    position = flag.start if flag.start is not None else cue_position
    return _SEVERITY_RANK.get(flag.severity, 9), position, cue_id, original_index


def _sorted_style_issues(issues: list[StyleIssue]) -> list[StyleIssue]:
    return sorted(issues, key=lambda issue: _SEVERITY_RANK.get(issue.severity, 9))


def _severity_counts(items: list[QCFlag] | list[StyleIssue]) -> dict[str, int]:
    return {
        severity: sum(1 for item in items if item.severity == severity)
        for severity in ("error", "warning", "info")
    }


def _format_seconds(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.3f}"
    return ""


def _format_multiline(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value)).replace("\n", "<br>")


def cue_time_label(cue: Cue) -> str:
    return f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}"


def _flag_seconds_to_ms(value: float | None) -> int:
    if value is None:
        return 0
    return max(0, int(round(value * 1000)))
