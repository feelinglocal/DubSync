from __future__ import annotations

import html
import json
from pathlib import Path

from .models import Cue, CueScore, QCFlag, StyleIssue
from .srt_io import format_timestamp, write_srt


def write_qc_report(
    report_json_path: Path,
    report_html_path: Path,
    cues: list[Cue],
    flags: list[QCFlag],
    style_issues: list[StyleIssue],
    cue_scores: list[CueScore] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "summary": {
            "cue_count": len(cues),
            "flags": len(flags),
            "style_violations": len(style_issues),
        },
        "cue_scores": [score.model_dump() for score in cue_scores or []],
        "flags": [flag.model_dump() for flag in flags],
        "style_issues": [issue.model_dump() for issue in style_issues],
    }
    report_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_html_path.write_text(_render_html(payload), encoding="utf-8")
    return payload


def write_changes_diff(path: Path, flags: list[QCFlag]) -> None:
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
        cues.append(
            Cue(
                index=len(cues) + 1,
                start_ms=_flag_seconds_to_ms(flag.start),
                end_ms=max(_flag_seconds_to_ms(flag.start), _flag_seconds_to_ms(flag.end)),
                lines=lines,
            )
        )
    path.write_text(write_srt(cues, renumber=True) if cues else "", encoding="utf-8")


def _render_html(payload: dict[str, object]) -> str:
    flags = payload.get("flags", [])
    issues = payload.get("style_issues", [])
    cue_scores = payload.get("cue_scores", [])
    score_rows = []
    for item in cue_scores if isinstance(cue_scores, list) else []:
        score_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('cue_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('cps', '')))}</td>"
            f"<td>{html.escape(str(item.get('score', '')))}</td>"
            f"<td>{html.escape(str(item.get('source', '')))}</td>"
            "</tr>"
        )
    rows = []
    for item in flags if isinstance(flags, list) else []:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('kind', '')))}</td>"
            f"<td>{html.escape(str(item.get('cue_ids', '')))}</td>"
            f"<td>{_format_seconds(item.get('start'))}</td>"
            f"<td>{_format_seconds(item.get('end'))}</td>"
            f"<td>{html.escape(str(item.get('message', '')))}</td>"
            f"<td>{html.escape(str(item.get('confidence', '')))}</td>"
            f"<td>{_format_multiline(item.get('old_text'))}</td>"
            f"<td>{_format_multiline(item.get('new_text'))}</td>"
            "</tr>"
        )
    issue_rows = []
    for item in issues if isinstance(issues, list) else []:
        issue_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('kind', '')))}</td>"
            f"<td>{html.escape(str(item.get('cue_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('message', '')))}</td>"
            "</tr>"
        )
    summary = payload.get("summary", {})
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>DubSync QC Report</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:32px;}"
        "table{border-collapse:collapse;width:100%;}td,th{border:1px solid #ccc;padding:6px;}"
        "th{background:#f2f2f2;text-align:left;}</style></head><body>"
        "<h1>DubSync QC Report</h1>"
        f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
        "<h2>Cue Scores</h2><table><tr><th>Cue</th><th>CPS</th><th>Score</th><th>Source</th></tr>"
        + "".join(score_rows)
        + "</table>"
        "<h2>Flags</h2><table><tr><th>Kind</th><th>Cues</th><th>Start</th><th>End</th>"
        "<th>Message</th><th>Confidence</th><th>Old Text</th><th>New Text</th></tr>"
        + "".join(rows)
        + "</table><h2>Style Issues</h2><table><tr><th>Kind</th><th>Cue</th><th>Message</th></tr>"
        + "".join(issue_rows)
        + "</table></body></html>"
    )


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
