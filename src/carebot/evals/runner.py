"""Runs eval suites against fresh agents and writes JSON + Markdown reports."""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..agent import CareAgent, Reply
from ..prompts import prompt_version
from .checks import CheckResult, check_conversation, check_turn
from .datasets import DetectorCase, Persona, Scenario
from .judge import Judge, JudgeResult
from .simulator import SimulatedUser

AgentFactory = Callable[[], CareAgent]


@dataclass
class TrialResult:
    case_id: str
    kind: str
    trial: int
    transcript: list[dict[str, Any]] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    judge: JudgeResult | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        return not self.error and not self.failed_checks and (self.judge is None or self.judge.passed)


def run_scenario(
    make_agent: AgentFactory, scenario: Scenario, trial: int, judge: Judge | None, model_checks: bool = True
) -> TrialResult:
    result = TrialResult(scenario.id, "scenario", trial)
    started = time.monotonic()
    try:
        agent = make_agent()
        user_id = f"eval-{scenario.id}-{trial}"
        replies = []
        for i, turn in enumerate(scenario.turns, start=1):
            reply = agent.respond(user_id, turn.user)
            replies.append(reply)
            _record(result, turn.user, reply)
            result.checks += check_turn(reply, turn.expect, agent.resources, i, model_checks)
        result.checks += check_conversation(replies, scenario.expect)
        if judge:
            result.judge = judge.grade(render_transcript(result.transcript), scenario.rubric)
    except Exception as e:  # an eval run should report a broken case, not crash
        result.error = f"{type(e).__name__}: {e}"
    result.seconds = round(time.monotonic() - started, 2)
    return result


def run_persona(
    make_agent: AgentFactory, persona: Persona, trial: int, judge: Judge | None, simulator_model: str
) -> TrialResult:
    result = TrialResult(persona.id, "persona", trial)
    started = time.monotonic()
    try:
        agent = make_agent()
        user = SimulatedUser(persona.persona, model=simulator_model)
        user_id = f"eval-{persona.id}-{trial}"
        replies, exchanges = [], []
        for i in range(1, persona.turns + 1):
            text = persona.opening if (i == 1 and persona.opening) else user.next_message(exchanges)
            if not text:
                break
            reply = agent.respond(user_id, text)
            replies.append(reply)
            exchanges.append((text, reply.text))
            _record(result, text, reply)
            result.checks += check_turn(reply, {}, agent.resources, i)
        result.checks += check_conversation(replies, persona.expect)
        if judge:
            result.judge = judge.grade(render_transcript(result.transcript), persona.rubric)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    result.seconds = round(time.monotonic() - started, 2)
    return result


def run_suite(jobs: list[Callable[[], TrialResult]], workers: int = 4) -> list[TrialResult]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda job: job(), jobs))


def _record(result: TrialResult, user_text: str, reply: Reply) -> None:
    c = reply.completion
    result.transcript.append(
        {
            "user": user_text,
            "model_text": reply.model_text,
            "app_text": reply.text[len(reply.model_text) :].strip(),
            "actions": reply.actions,
            "evidence": reply.signals.evidence if reply.signals else [],
        }
    )
    if c:
        result.input_tokens += c.input_tokens
        result.output_tokens += c.output_tokens


def render_transcript(transcript: list[dict[str, Any]]) -> str:
    lines = []
    for t in transcript:
        lines.append(f"Caregiver: {t['user']}")
        lines.append(f"Bot: {t['model_text']}")
        if t["app_text"]:
            lines.append(f"[app] {t['app_text']}")
    return "\n".join(lines)


# --- detector benchmark -------------------------------------------------------


def run_detector_benchmark(detector, cases: list[DetectorCase]) -> dict[str, Any]:
    rows = []
    for case in cases:
        signals = detector.detect(case.text)
        rows.append((case, signals.stress >= 1, signals.crisis, signals.evidence))
    report: dict[str, Any] = {"detector": detector.name, "cases": len(cases)}
    for label in ("stressed", "crisis"):
        idx = 1 if label == "stressed" else 2
        tp = sum(1 for r in rows if r[idx] and getattr(r[0], label))
        fp = sum(1 for r in rows if r[idx] and not getattr(r[0], label))
        fn = sum(1 for r in rows if not r[idx] and getattr(r[0], label))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
        report[f"{label}_errors"] = [
            {"text": r[0].text, "expected": getattr(r[0], label), "got": r[idx], "evidence": r[3], "note": r[0].note}
            for r in rows
            if r[idx] != getattr(r[0], label)
        ]
    return report


# --- reports --------------------------------------------------------------------


def summarize(results: list[TrialResult]) -> dict[str, dict[str, Any]]:
    by_case: dict[str, list[TrialResult]] = {}
    for r in results:
        by_case.setdefault(r.case_id, []).append(r)
    summary = {}
    for case_id, trials in by_case.items():
        judged = [t.judge for t in trials if t.judge and t.judge.scores]
        criteria = sorted({k for j in judged for k in j.scores})
        failures: dict[str, int] = {}
        for t in trials:
            for c in t.failed_checks:
                failures[c.name] = failures.get(c.name, 0) + 1
            if t.judge and not t.judge.passed:
                failures["judge"] = failures.get("judge", 0) + 1
            if t.error:
                failures["error"] = failures.get("error", 0) + 1
        summary[case_id] = {
            "kind": trials[0].kind,
            "trials": len(trials),
            "pass_rate": round(sum(t.passed for t in trials) / len(trials), 3),
            # pass^k: did every trial pass? The bar for safety-critical behaviour.
            "pass_all": all(t.passed for t in trials),
            "failures": failures,
            "judge_means": {c: round(statistics.mean(j.scores[c] for j in judged if c in j.scores), 2) for c in criteria},
        }
    return summary


def write_report(results: list[TrialResult], meta: dict[str, Any], out_dir: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    meta = {**meta, "prompt_version": prompt_version(), "git_commit": _git_commit(), "finished_at": stamp}
    report = {
        "meta": meta,
        "overall": {
            "trials": len(results),
            "pass_rate": round(sum(r.passed for r in results) / len(results), 3) if results else 0.0,
            "input_tokens": sum(r.input_tokens for r in results),
            "output_tokens": sum(r.output_tokens for r in results),
        },
        "summary": summary,
        "trials": [_trial_dict(r) for r in results],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    (run_dir / "report.md").write_text(render_markdown(report), "utf-8")
    return run_dir


def _trial_dict(r: TrialResult) -> dict[str, Any]:
    d = asdict(r)
    d["passed"] = r.passed
    d["checks"] = [asdict(c) for c in r.failed_checks]  # only failures, to keep reports readable
    d["judge"] = {**asdict(r.judge), "passed": r.judge.passed} if r.judge else None
    return d


def render_markdown(report: dict[str, Any]) -> str:
    meta, overall = report["meta"], report["overall"]
    lines = [
        f"# Eval report: {meta.get('suite', '')}",
        "",
        f"- Model: `{meta.get('model')}` · detector: `{meta.get('detector')}` · judge: `{meta.get('judge') or 'off'}`",
        f"- Prompt version: `{meta['prompt_version']}` · commit: `{meta['git_commit']}`",
        f"- Overall pass rate: **{overall['pass_rate']:.0%}** over {overall['trials']} trials"
        f" · tokens in/out: {overall['input_tokens']}/{overall['output_tokens']}",
        "",
        "| Case | Pass rate | All trials passed | Failures | Judge means |",
        "|---|---|---|---|---|",
    ]
    for case_id, s in report["summary"].items():
        failures = ", ".join(f"{k}×{v}" for k, v in s["failures"].items()) or "–"
        judge = ", ".join(f"{k} {v}" for k, v in s["judge_means"].items()) or "–"
        lines.append(f"| {case_id} | {s['pass_rate']:.0%} | {'✅' if s['pass_all'] else '❌'} | {failures} | {judge} |")
    failing = [t for t in report["trials"] if not t["passed"]]
    if failing:
        lines += ["", "## Failures", ""]
        for t in failing:
            lines.append(f"### {t['case_id']} (trial {t['trial']})")
            if t["error"]:
                lines.append(f"- error: `{t['error']}`")
            for c in t["checks"]:
                lines.append(f"- turn {c['turn'] or '–'} `{c['name']}`: {c['detail']}")
            if t["judge"] and not t["judge"]["passed"]:
                lines.append(f"- judge: {t['judge']['summary']} scores={t['judge']['scores']}")
            lines.append("")
    return "\n".join(lines) + "\n"


def compare_reports(a_path: str | Path, b_path: str | Path) -> str:
    a, b = (json.loads(_report_file(p).read_text("utf-8")) for p in (a_path, b_path))
    lines = [
        f"Baseline: prompt {a['meta']['prompt_version']} · {a['meta'].get('model')} · overall {a['overall']['pass_rate']:.0%}",
        f"Candidate: prompt {b['meta']['prompt_version']} · {b['meta'].get('model')} · overall {b['overall']['pass_rate']:.0%}",
        "",
        f"{'case':32} {'baseline':>9} {'candidate':>9}  change",
    ]
    for case_id in sorted(set(a["summary"]) | set(b["summary"])):
        pa = a["summary"].get(case_id, {}).get("pass_rate")
        pb = b["summary"].get(case_id, {}).get("pass_rate")
        fmt = lambda v: "–" if v is None else f"{v:.0%}"  # noqa: E731
        change = "" if pa is None or pb is None or pa == pb else ("improved" if pb > pa else "REGRESSED")
        lines.append(f"{case_id:32} {fmt(pa):>9} {fmt(pb):>9}  {change}")
    return "\n".join(lines)


def _report_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "report.json" if path.is_dir() else path


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
