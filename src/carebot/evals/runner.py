"""Runs eval suites against fresh agents and writes JSON + Markdown reports."""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..agent import CareAgent, Reply
from ..prompts import prompt_version
from .checks import CheckResult, check_conversation, check_turn
from .datasets import DetectorCase, Persona, RecommendationCase, Scenario
from .judge import Judge, JudgeResult
from .simulator import SimulatedUser
from .stats import bootstrap_ci, bootstrap_diff_ci, mean

AgentFactory = Callable[[], CareAgent]


@dataclass
class TrialResult:
    case_id: str
    kind: str
    trial: int
    tags: tuple[str, ...] = ()
    transcript: list[dict[str, Any]] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    judge: JudgeResult | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
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
    result = TrialResult(scenario.id, "scenario", trial, scenario.tags)
    started = time.monotonic()
    try:
        agent = make_agent()
        user_id = f"eval-{scenario.id}-{trial}"
        replies = []
        for i, turn in enumerate(scenario.turns, start=1):
            if turn.choose:
                reply = agent.choose(user_id, turn.choose)
                _record(result, f"[taps {turn.choose}]", reply)
            else:
                reply = agent.respond(user_id, turn.user)
                _record(result, turn.user, reply)
            replies.append(reply)
            result.checks += check_turn(reply, turn.expect, agent.resources, i, model_checks, agent.mode)
        result.checks += check_conversation(replies, scenario.expect, agent.mode)
        if judge:
            result.judge = judge.grade(render_transcript(result.transcript), scenario.rubric, tools=agent.mode == "tools")
    except Exception as e:  # an eval run should report a broken case, not crash
        result.error = f"{type(e).__name__}: {e}"
    result.seconds = round(time.monotonic() - started, 2)
    return result


def run_persona(
    make_agent: AgentFactory, persona: Persona, trial: int, judge: Judge | None, simulator_model: str
) -> TrialResult:
    result = TrialResult(persona.id, "persona", trial, persona.tags)
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
            _record(result, text, reply)
            result.checks += check_turn(reply, {}, agent.resources, i, agent_mode=agent.mode)
            seen = reply.text
            if reply.buttons and persona.accepts_offers:
                # The simulated caregiver taps "Yes" and reads the services before replying.
                chosen = agent.choose(user_id, "yes")
                replies.append(chosen)
                _record(result, "[taps yes]", chosen)
                seen += f"\n\n[You tapped Yes]\n\n{chosen.text}"
            exchanges.append((text, seen))
        result.checks += check_conversation(replies, persona.expect, agent.mode)
        if judge:
            result.judge = judge.grade(render_transcript(result.transcript), persona.rubric, tools=agent.mode == "tools")
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    result.seconds = round(time.monotonic() - started, 2)
    return result


def run_suite(jobs: list[Callable[[], TrialResult]], workers: int = 4) -> list[TrialResult]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda job: job(), jobs))


def _record(result: TrialResult, user_text: str, reply: Reply) -> None:
    result.transcript.append(
        {
            "user": user_text,
            "model_text": reply.model_text,
            "app_text": reply.text[len(reply.model_text) :].strip(),
            "actions": reply.actions,
            "trajectory": reply.trajectory,
            "escalation": reply.escalation,
            "recommended": reply.recommended,
            "evidence": reply.signals.evidence if reply.signals else [],
            "latency_ms": reply.latency_ms,
        }
    )
    if reply.completion:
        result.input_tokens += reply.completion.input_tokens
        result.output_tokens += reply.completion.output_tokens
    result.latency_ms += reply.latency_ms


def render_transcript(transcript: list[dict[str, Any]]) -> str:
    lines = []
    for t in transcript:
        lines.append(f"Caregiver: {t['user']}")
        for step in t.get("trajectory", []):
            lines.append(f"[tool] {step['tool']}({json.dumps(step['input'], ensure_ascii=False)}) -> {step['result'][:200]}")
        if t["model_text"]:
            lines.append(f"Bot: {t['model_text']}")
        if t["app_text"]:
            lines.append(f"[app] {t['app_text']}")
    return "\n".join(lines)


# --- single-component benchmarks ---------------------------------------------------


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


def run_retrieval_benchmark(retriever, cases: list[RecommendationCase], k: int = 3) -> dict[str, Any]:
    """Recommendation accuracy: is a service that would actually help near the top?"""
    hits1, recalls, reciprocal, rows = [], [], [], []
    for case in cases:
        ranked = [c.id for c, score in retriever.scores(" ".join(case.messages)) if score > 0]
        top = retriever.search(" ".join(case.messages), k)
        top_ids = [c.id for c in top]
        relevant = set(case.relevant)
        hits1.append(float(bool(top_ids) and top_ids[0] in relevant))
        recalls.append(len(relevant & set(top_ids)) / len(relevant))
        rank = next((i for i, cid in enumerate(ranked, start=1) if cid in relevant), None)
        reciprocal.append(1 / rank if rank else 0.0)
        rows.append({"id": case.id, "relevant": sorted(relevant), "top": top_ids, "first_relevant_rank": rank})
    return {
        "cases": len(cases),
        "k": k,
        "hit@1": round(mean(hits1), 3),
        f"recall@{k}": round(mean(recalls), 3),
        "mrr": round(mean(reciprocal), 3),
        "misses": [r for r, h in zip(rows, hits1) if not h],
    }


# --- reports --------------------------------------------------------------------


def summarize(results: list[TrialResult]) -> dict[str, dict[str, Any]]:
    by_case: dict[str, list[TrialResult]] = {}
    for r in results:
        by_case.setdefault(r.case_id, []).append(r)
    summary = {}
    for case_id, trials in by_case.items():
        judged = [t.judge for t in trials if t.judge and t.judge.scores]
        criteria = sorted({k for j in judged for k in j.scores})
        failures: Counter[str] = Counter()
        for t in trials:
            failures.update(c.name for c in t.failed_checks)
            if t.judge and not t.judge.passed:
                failures["judge"] += 1
            if t.error:
                failures["error"] += 1
        turns = [turn for t in trials for turn in t.transcript]
        summary[case_id] = {
            "kind": trials[0].kind,
            "tags": list(trials[0].tags),
            "trials": len(trials),
            "pass_rate": round(mean([t.passed for t in trials]), 3),
            # pass^k: did every trial pass? The bar for safety-critical behaviour.
            "pass_all": all(t.passed for t in trials),
            "failures": dict(failures),
            "judge_means": {c: round(mean([j.scores[c] for j in judged if c in j.scores]), 2) for c in criteria},
            "escalations": dict(Counter(turn["escalation"] for turn in turns if turn.get("escalation"))),
            "tool_calls": dict(Counter(step["tool"] for turn in turns for step in turn.get("trajectory", []))),
            "mean_latency_ms": round(mean([t.latency_ms for t in trials])),
        }
    return summary


def overall_metrics(results: list[TrialResult]) -> dict[str, Any]:
    passes = [float(r.passed) for r in results]
    low, high = bootstrap_ci(passes)
    turns = [turn for r in results for turn in r.transcript]
    escalations = Counter(turn["escalation"] for turn in turns if turn.get("escalation"))
    total_escalations = sum(escalations.values())
    by_model = escalations["model"] + escalations["both"]
    return {
        "trials": len(results),
        "pass_rate": round(mean(passes), 3) if results else 0.0,
        "pass_rate_ci95": [round(low, 3), round(high, 3)],
        "safety_pass_all": all(r.passed for r in results if "safety" in r.tags) if results else True,
        # Share of crisis turns where the model escalated by itself (tools mode),
        # rather than the harness's crisis screen having to do it.
        "self_escalation_rate": round(by_model / total_escalations, 3) if total_escalations else None,
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "mean_latency_ms_per_turn": round(mean([t["latency_ms"] for t in turns])) if turns else 0,
    }


def write_report(results: list[TrialResult], meta: dict[str, Any], out_dir: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / f"{stamp}-{meta.get('suite', 'eval')}-{meta.get('label', 'run')}".replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "prompt_version": prompt_version(), "git_commit": _git_commit(), "finished_at": stamp}
    report = {
        "meta": meta,
        "overall": overall_metrics(results),
        "summary": summarize(results),
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
    low, high = overall["pass_rate_ci95"]
    lines = [
        f"# Eval report: {meta.get('suite', '')}",
        "",
        f"- Model: `{meta.get('model')}` · agent: `{meta.get('agent')}` · detector: `{meta.get('detector')}`"
        f" · judge: `{meta.get('judge') or 'off'}`",
        f"- Prompt version: `{meta['prompt_version']}` · test set: `{meta.get('dataset_version')}`"
        f" · commit: `{meta['git_commit']}`",
        f"- Overall pass rate: **{overall['pass_rate']:.0%}** (95% CI {low:.0%}–{high:.0%}) over {overall['trials']} trials"
        f" · safety cases all passed: {'yes' if overall['safety_pass_all'] else '**no**'}",
        f"- Tokens in/out: {overall['input_tokens']}/{overall['output_tokens']}"
        f" · mean latency per turn: {overall['mean_latency_ms_per_turn']} ms",
    ]
    if meta.get("agent") == "tools" and overall["self_escalation_rate"] is not None:
        lines.append(f"- Crisis turns escalated by the model itself: {overall['self_escalation_rate']:.0%}")
    lines += [
        "",
        "| Case | Pass rate | All trials passed | Failures | Judge means | Tools used |",
        "|---|---|---|---|---|---|",
    ]
    for case_id, s in report["summary"].items():
        failures = ", ".join(f"{k}×{v}" for k, v in s["failures"].items()) or "–"
        judge = ", ".join(f"{k} {v}" for k, v in s["judge_means"].items()) or "–"
        tools = ", ".join(f"{k}×{v}" for k, v in s["tool_calls"].items()) or "–"
        lines.append(
            f"| {case_id} | {s['pass_rate']:.0%} | {'✅' if s['pass_all'] else '❌'} | {failures} | {judge} | {tools} |"
        )
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
    """Per-case pass-rate change with a bootstrap interval, so noise isn't read as a regression."""
    a, b = (json.loads(report_file(p).read_text("utf-8")) for p in (a_path, b_path))
    lines = []
    if a["meta"].get("dataset_version") != b["meta"].get("dataset_version"):
        lines += ["WARNING: the two runs used different eval files; differences may come from the test set.", ""]
    for name, r in (("Baseline", a), ("Candidate", b)):
        m = r["meta"]
        lines.append(
            f"{name}: {m.get('model')} · agent {m.get('agent')} · detector {m.get('detector')} · prompt {m['prompt_version']}"
            f" · overall {r['overall']['pass_rate']:.0%}"
        )
    lines += ["", f"{'case':32} {'baseline':>9} {'candidate':>9}  {'change [95% CI]':24} verdict"]
    passes_a, passes_b = _passes_by_case(a), _passes_by_case(b)
    for case_id in sorted(set(passes_a) | set(passes_b)) + ["OVERALL"]:
        pa = sum(passes_a.values(), []) if case_id == "OVERALL" else passes_a.get(case_id)
        pb = sum(passes_b.values(), []) if case_id == "OVERALL" else passes_b.get(case_id)
        if not pa or not pb:
            lines.append(f"{case_id:32} {_pct(pa):>9} {_pct(pb):>9}  (only in one run)")
            continue
        low, high = bootstrap_diff_ci(pa, pb)
        delta = mean(pb) - mean(pa)
        verdict = "REGRESSED" if high < 0 else "improved" if low > 0 else ("same" if delta == 0 else "within noise")
        change = f"{delta:+.0%} [{low:+.0%}, {high:+.0%}]"
        lines.append(f"{case_id:32} {_pct(pa):>9} {_pct(pb):>9}  {change:24} {verdict}")
    return "\n".join(lines)


def _passes_by_case(report: dict[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for t in report["trials"]:
        out.setdefault(t["case_id"], []).append(float(t["passed"]))
    return out


def _pct(values: list[float] | None) -> str:
    return "–" if not values else f"{mean(values):.0%}"


def render_benchmark(rows: list[tuple[str, dict[str, Any], Path]]) -> str:
    """One table comparing several configurations run on the same test set."""
    lines = [
        "# Benchmark",
        "",
        "| Configuration | Pass rate [95% CI] | Safety cases all passed | Self-escalation | Judge means | Tokens in/out | Latency/turn | Report |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, report, run_dir in rows:
        o = report["overall"]
        low, high = o["pass_rate_ci95"]
        judged = [s["judge_means"] for s in report["summary"].values() if s["judge_means"]]
        criteria = sorted({c for j in judged for c in j})
        judge = ", ".join(f"{c} {mean([j[c] for j in judged if c in j]):.1f}" for c in criteria) or "–"
        rate = o["self_escalation_rate"]
        self_esc = "–" if rate is None or report["meta"].get("agent") != "tools" else f"{rate:.0%}"
        lines.append(
            f"| {label} | {o['pass_rate']:.0%} [{low:.0%}, {high:.0%}] | {'yes' if o['safety_pass_all'] else 'no'} "
            f"| {self_esc} | {judge} | {o['input_tokens']}/{o['output_tokens']} | {o['mean_latency_ms_per_turn']} ms "
            f"| `{run_dir.name}` |"
        )
    return "\n".join(lines) + "\n"


def report_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "report.json" if path.is_dir() else path


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
