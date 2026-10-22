"""Human-in-the-loop evaluation: people rate conversations, and we measure the judge against them.

1. `carebot eval export-ratings REPORT` writes a blind rating sheet (CSV, opens
   in Excel or Google Sheets): one row per conversation, shuffled, without the
   judge's scores.
2. Each rater fills in 1-5 per criterion and saves their own copy.
3. `carebot eval agreement REPORT sheet1.csv [sheet2.csv ...]` reports how well
   the LLM judge agrees with each rater, and how well the raters agree with
   each other. Trust the judge on a criterion only where it agrees with people
   about as well as people agree with each other.
"""

from __future__ import annotations

import csv
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any

from .judge import DEFAULT_CRITERIA, PASS_SCORE
from .runner import render_transcript, report_file
from .stats import cohen_kappa, mean, spearman, weighted_kappa

FIXED_COLUMNS = ["item_id", "case_id", "trial", "conversation"]
TRAILING_COLUMNS = ["rater", "notes"]


def _criteria(report: dict[str, Any]) -> list[str]:
    found = []
    for t in report["trials"]:
        for name in (t.get("judge") or {}).get("scores", {}):
            if name not in found:
                found.append(name)
    return found or list(DEFAULT_CRITERIA)


def export_ratings(report_path: str | Path, out_path: str | Path, seed: int = 0) -> int:
    report = json.loads(report_file(report_path).read_text("utf-8"))
    criteria = _criteria(report)
    trials = list(report["trials"])
    random.Random(seed).shuffle(trials)  # so raters don't see cases in a meaningful order
    with Path(out_path).open("w", newline="", encoding="utf-8-sig") as f:  # -sig: Excel reads Chinese correctly
        writer = csv.writer(f)
        writer.writerow(FIXED_COLUMNS + criteria + TRAILING_COLUMNS)
        for t in trials:
            row = [f"{t['case_id']}#{t['trial']}", t["case_id"], t["trial"], render_transcript(t["transcript"])]
            writer.writerow(row + [""] * len(criteria) + ["", ""])
    return len(trials)


def _read_sheet(path: str | Path, criteria: list[str]) -> dict[str, dict[str, int]]:
    ratings = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            scores = {}
            for c in criteria:
                value = (row.get(c) or "").strip()
                if value:
                    scores[c] = min(max(int(float(value)), 1), 5)
            if scores:
                ratings[row["item_id"]] = scores
    return ratings


def agreement(report_path: str | Path, sheets: list[str | Path]) -> dict[str, Any]:
    report = json.loads(report_file(report_path).read_text("utf-8"))
    criteria = _criteria(report)
    judge = {
        f"{t['case_id']}#{t['trial']}": t["judge"]["scores"]
        for t in report["trials"]
        if t.get("judge") and t["judge"].get("scores")
    }
    raters = {Path(s).stem: _read_sheet(s, criteria) for s in sheets}
    result: dict[str, Any] = {"criteria": criteria, "judge_vs_human": {}, "human_vs_human": {}}
    if judge:
        for name, ratings in raters.items():
            result["judge_vs_human"][name] = _compare(judge, ratings, criteria)
    for (n1, r1), (n2, r2) in combinations(raters.items(), 2):
        result["human_vs_human"][f"{n1} vs {n2}"] = _compare(r1, r2, criteria)
    return result


def _compare(a: dict[str, dict[str, int]], b: dict[str, dict[str, int]], criteria: list[str]) -> dict[str, Any]:
    items = sorted(set(a) & set(b))
    out: dict[str, Any] = {"items": len(items), "by_criterion": {}}
    for c in criteria:
        pairs = [(a[i][c], b[i][c]) for i in items if c in a[i] and c in b[i]]
        if not pairs:
            continue
        x, y = [p[0] for p in pairs], [p[1] for p in pairs]
        out["by_criterion"][c] = {
            "n": len(pairs),
            "exact": round(mean([p == q for p, q in pairs]), 3),
            "within_1": round(mean([abs(p - q) <= 1 for p, q in pairs]), 3),
            "weighted_kappa": round(weighted_kappa(x, y), 3),
            "spearman": round(spearman(x, y), 3),
        }
    # Pass/fail on the whole conversation: every criterion at least PASS_SCORE.
    both = [i for i in items if set(criteria) <= set(a[i]) and set(criteria) <= set(b[i])]
    if both:
        pa = [all(a[i][c] >= PASS_SCORE for c in criteria) for i in both]
        pb = [all(b[i][c] >= PASS_SCORE for c in criteria) for i in both]
        out["pass_fail"] = {
            "n": len(both),
            "agreement": round(mean([p == q for p, q in zip(pa, pb)]), 3),
            "kappa": round(cohen_kappa(pa, pb), 3),
        }
    return out


def render_agreement(result: dict[str, Any]) -> str:
    lines = []
    for title, key in (("LLM judge vs each human rater", "judge_vs_human"), ("Human raters vs each other", "human_vs_human")):
        if not result[key]:
            continue
        lines += [f"## {title}", ""]
        for pair, stats in result[key].items():
            lines.append(f"### {pair} ({stats['items']} conversations)")
            lines.append("| Criterion | n | Exact | Within 1 | Weighted κ | Spearman |")
            lines.append("|---|---|---|---|---|---|")
            for c, m in stats["by_criterion"].items():
                lines.append(
                    f"| {c} | {m['n']} | {m['exact']:.0%} | {m['within_1']:.0%} | {m['weighted_kappa']:.2f} | {m['spearman']:.2f} |"
                )
            if "pass_fail" in stats:
                pf = stats["pass_fail"]
                lines.append(f"\nPass/fail agreement: {pf['agreement']:.0%} (κ {pf['kappa']:.2f}, n={pf['n']})")
            lines.append("")
    if not lines:
        return "Nothing to compare: the report has no judge scores and only one rating sheet was given.\n"
    lines.append("Rough guide to κ: <0.2 poor, 0.2–0.4 fair, 0.4–0.6 moderate, 0.6–0.8 substantial, >0.8 near-perfect.")
    return "\n".join(lines) + "\n"
