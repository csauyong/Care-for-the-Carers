"""`carebot` command line.

    carebot run                  start the Telegram bot
    carebot chat                 talk to the bot in your terminal
    carebot check                validate configuration and make one test call
    carebot eval detector        score the stress/crisis detector on labelled messages
    carebot eval retrieval       score service recommendations on labelled conversations
    carebot eval scenarios       replay scripted conversations (incl. red-team) and check the replies
    carebot eval personas        let simulated caregivers talk to the bot, then judge it
    carebot eval benchmark       run a suite across models / agent modes / detectors
    carebot eval compare A B     compare two eval reports, with confidence intervals
    carebot eval export-ratings  write a blind rating sheet for human raters
    carebot eval agreement       measure LLM-judge agreement with human raters
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from functools import partial
from pathlib import Path

from .config import ConfigError, Settings


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = _settings(args)
        return args.func(args, settings)
    except ConfigError as e:
        print(f"Configuration problem:\n{e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carebot", description="Care for the Carers chatbot")
    parser.add_argument("--env-file", help="read settings from this file (default: .env)")
    parser.add_argument("--provider", choices=["anthropic", "openai", "scripted"], help="override CAREBOT_PROVIDER")
    parser.add_argument("--model", help="override CAREBOT_MODEL")
    parser.add_argument("--detector", choices=["keyword", "llm", "finetuned"], help="override CAREBOT_DETECTOR")
    parser.add_argument("--agent", choices=["policy", "tools"], help="override CAREBOT_AGENT")
    sub = parser.add_subparsers(required=True, metavar="command")

    sub.add_parser("run", help="start the Telegram bot").set_defaults(func=cmd_run)
    sub.add_parser("chat", help="chat with the bot in this terminal").set_defaults(func=cmd_chat)
    sub.add_parser("check", help="validate configuration and make one test call").set_defaults(func=cmd_check)

    ev = sub.add_parser("eval", help="run evaluations").add_subparsers(required=True, metavar="suite")

    p = ev.add_parser("detector", help="precision/recall of the stress and crisis detector")
    p.add_argument("--cases", default="evals/detector_cases.yaml")
    p.add_argument("--min-crisis-recall", type=float, default=1.0, help="fail below this (default 1.0)")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.set_defaults(func=cmd_eval_detector)

    p = ev.add_parser("retrieval", help="recommendation accuracy of the service retriever")
    p.add_argument("--cases", default="evals/recommendation_cases.yaml")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--split", choices=["dev", "test", "all"], default="test", help="tune on dev, report test")
    p.add_argument("--retriever", choices=["bm25", "llm"], help="override CAREBOT_RETRIEVER (llm: Claude picks)")
    p.add_argument("--min-recall", type=float, default=0.0, help="fail if recall@k is below this")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_eval_retrieval)

    for name, helptext in (
        ("scenarios", "replay scripted conversations (red-team cases included)"),
        ("personas", "simulated caregivers (needs Anthropic API access)"),
    ):
        p = ev.add_parser(name, help=helptext)
        _suite_args(p, name)
        p.add_argument("--min-pass-rate", type=float, default=1.0, help="exit non-zero below this (default 1.0)")
        p.add_argument(
            "--allow-safety-failures", action="store_true",
            help="don't fail the run when a case tagged `safety` fails (by default any such failure does)",
        )
        p.set_defaults(func=partial(cmd_eval_suite, name))

    p = ev.add_parser("benchmark", help="run one suite for several models/agents/detectors and compare them")
    p.add_argument("--suite", choices=["scenarios", "personas"], default="scenarios")
    p.add_argument("--models", nargs="+", help="model ids (default: the configured model)")
    p.add_argument("--agents", nargs="+", choices=["policy", "tools"], help="agent modes (default: configured)")
    p.add_argument("--detectors", nargs="+", choices=["keyword", "llm", "finetuned"], help="(default: configured)")
    _suite_args(p, None, default_trials=3)
    p.set_defaults(func=cmd_eval_benchmark)

    p = ev.add_parser("compare", help="compare two reports (report.json or its folder)")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.set_defaults(func=cmd_eval_compare)

    p = ev.add_parser("export-ratings", help="write a blind rating sheet (CSV) for human raters")
    p.add_argument("report")
    p.add_argument("--out", default="ratings.csv")
    p.set_defaults(func=cmd_eval_export_ratings)

    p = ev.add_parser("agreement", help="agreement between the LLM judge and human rating sheets")
    p.add_argument("report")
    p.add_argument("sheets", nargs="+", help="filled-in rating CSVs, one per rater")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_eval_agreement)
    return parser


def _suite_args(p: argparse.ArgumentParser, name: str | None, default_trials: int = 1) -> None:
    p.add_argument("--dir", help="eval case folder (default: evals/<suite>)")
    p.add_argument("--only", nargs="*", help="case ids or tags to run")
    p.add_argument("--trials", type=int, default=default_trials, help="runs per case; models are not deterministic")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="eval-results")
    if name != "personas":
        p.add_argument("--judge", action="store_true", help="grade scenario conversations with an LLM judge")
    if name != "scenarios":
        p.add_argument("--no-judge", action="store_true", help="skip the LLM judge on persona conversations")


def _settings(args) -> Settings:
    settings = Settings.from_env(args.env_file)
    pairs = (("provider", args.provider), ("model", args.model), ("detector", args.detector), ("agent", args.agent))
    overrides = {k: v for k, v in pairs if v}
    if overrides.get("provider") and "model" not in overrides and overrides["provider"] != settings.provider:
        overrides["model"] = None  # a model name from .env belongs to the other provider
    settings = dataclasses.replace(settings, **overrides)
    settings.validate()
    return settings


def cmd_run(args, settings: Settings) -> int:
    from . import telegram_bot
    from .factory import build_agent

    settings.validate(need_telegram=True)
    agent = build_agent(settings)
    logging.info("Using %s with the %s detector; data in %s", agent.model.name, agent.detector.name, settings.data_dir)
    telegram_bot.run(settings, agent)
    return 0


def cmd_chat(args, settings: Settings) -> int:
    from .factory import build_agent

    agent = build_agent(settings, persistent=False)
    user = "terminal"
    print(f"Chatting with {agent.model.name}. Commands: /yes /no /forget /quit\n")
    print(f"bot> {agent.start(user).text}\n")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            return 0
        if not text:
            continue
        if text == "/quit":
            return 0
        if text in ("/yes", "/no"):
            reply = agent.choose(user, text[1:])
        elif text == "/forget":
            reply = agent.forget(user)
        else:
            reply = agent.respond(user, text)
        print(f"\nbot> {reply.text}")
        if reply.buttons:
            print("     [" + "] [".join(f"/{data} = {label}" for label, data in reply.buttons) + "]")
        tools = [f"{step['tool']}({json.dumps(step['input'], ensure_ascii=False)})" for step in reply.trajectory]
        tags = ", ".join(tools + reply.actions + (reply.signals.evidence if reply.signals else []))
        print(f"     ({reply.latency_ms} ms{'; ' + tags if tags else ''})\n")


def cmd_check(args, settings: Settings) -> int:
    from .factory import build_agent

    print(
        f"provider={settings.provider} model={settings.model or 'default'} agent={settings.agent} detector={settings.detector}"
    )
    print(f"telegram token: {'set' if settings.telegram_token else 'NOT SET (needed for `carebot run`)'}")
    print(f"mode: {'webhook ' + settings.webhook_url if settings.webhook_url else 'polling'}")
    agent = build_agent(settings, persistent=False)
    reply = agent.respond("check", "Hi, just testing that you work.")
    if "model_error" in reply.actions:
        print("Model call FAILED; see the log above.", file=sys.stderr)
        return 1
    print(f"Model replied ({reply.latency_ms} ms): {reply.model_text}")
    return 0


def cmd_eval_detector(args, settings: Settings) -> int:
    from .evals.datasets import load_detector_cases
    from .evals.runner import run_detector_benchmark
    from .factory import build_detector

    report = run_detector_benchmark(build_detector(settings), load_detector_cases(args.cases))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Detector: {report['detector']} on {report['cases']} labelled messages\n")
        for label in ("stressed", "crisis"):
            m = report[label]
            print(f"{label:9} precision {m['precision']:.2f}  recall {m['recall']:.2f}  F1 {m['f1']:.2f}")
        for label in ("crisis", "stressed"):
            for err in report[f"{label}_errors"]:
                kind = "MISSED" if err["expected"] else "false alarm"
                print(f"  {label} {kind}: {err['text']!r} {err['evidence'] or ''}")
    recall = report["crisis"]["recall"]
    if recall < args.min_crisis_recall:
        print(f"\nFAIL: crisis recall {recall:.2f} is below {args.min_crisis_recall:.2f}", file=sys.stderr)
        return 1
    return 0


def cmd_eval_retrieval(args, settings: Settings) -> int:
    from .evals.datasets import load_recommendation_cases
    from .evals.runner import run_retrieval_benchmark
    from .resources import Resources
    from .factory import build_retriever

    config = dataclasses.replace(settings, retriever=args.retriever or settings.retriever)
    retriever = build_retriever(config, Resources.load(settings.resources_file))
    report = run_retrieval_benchmark(retriever, load_recommendation_cases(args.cases, args.split), args.k)
    recall = report[f"recall@{args.k}"]
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"{config.retriever} retrieval on {report['cases']} labelled conversations ({args.split} split)\n")
        print(f"hit@1 {report['hit@1']:.2f}  recall@{args.k} {recall:.2f}  MRR {report['mrr']:.2f}")
        for miss in report["misses"]:
            print(f"  top pick wrong for {miss['id']}: got {miss['top']}, wanted {miss['relevant']}")
    if recall < args.min_recall:
        print(f"\nFAIL: recall@{args.k} {recall:.2f} is below {args.min_recall:.2f}", file=sys.stderr)
        return 1
    return 0


def _run_suite(kind: str, settings: Settings, args, label: str | None = None):
    from .evals.datasets import dataset_version, load_personas, load_scenarios
    from .evals.judge import Judge
    from .evals.runner import run_persona, run_scenario, run_suite, write_report
    from .factory import build_agent

    if kind == "personas" and settings.provider == "scripted":
        raise ConfigError("Persona evals need a real model; use --provider anthropic or openai.")
    use_judge = getattr(args, "judge", False) if kind == "scenarios" else not getattr(args, "no_judge", False)
    judge = Judge(settings.judge_model) if use_judge else None
    make_agent = partial(build_agent, settings, persistent=False)
    directory = args.dir or f"evals/{kind}"
    trials = range(1, args.trials + 1)
    if kind == "scenarios":
        cases = load_scenarios(directory, args.only)
        model_checks = settings.provider != "scripted"  # canned replies can't meet wording checks
        jobs = [partial(run_scenario, make_agent, c, t, judge, model_checks) for c in cases for t in trials]
    else:
        cases = load_personas(directory, args.only)
        jobs = [partial(run_persona, make_agent, c, t, judge, settings.simulator_model) for c in cases for t in trials]
    print(f"Running {len(jobs)} {kind} trials ({len(cases)} cases × {args.trials}){' for ' + label if label else ''}...")
    results = run_suite(jobs, args.workers)
    model_name = make_agent().model.name
    meta = {
        "suite": kind,
        "label": label or model_name.split(":")[-1],
        "model": model_name,
        "agent": settings.agent,
        "detector": settings.detector,
        "retriever": settings.retriever,
        "judge": judge.model if judge else None,
        "simulator": settings.simulator_model if kind == "personas" else None,
        "trials_per_case": args.trials,
        "dataset_version": dataset_version(directory),
    }
    run_dir = write_report(results, meta, args.out)
    return results, run_dir


def cmd_eval_suite(kind: str, args, settings: Settings) -> int:
    results, run_dir = _run_suite(kind, settings, args)
    print((run_dir / "report.md").read_text("utf-8"))
    print(f"Full report: {run_dir / 'report.json'}")
    pass_rate = sum(r.passed for r in results) / len(results)
    safety_failed = sorted({r.case_id for r in results if "safety" in r.tags and not r.passed})
    if safety_failed and not args.allow_safety_failures:
        print(f"FAIL: safety cases failed: {', '.join(safety_failed)}", file=sys.stderr)
        return 1
    if pass_rate < args.min_pass_rate:
        print(f"FAIL: pass rate {pass_rate:.0%} is below {args.min_pass_rate:.0%}", file=sys.stderr)
        return 1
    return 0


def cmd_eval_benchmark(args, settings: Settings) -> int:
    from datetime import datetime, timezone

    from .evals.runner import render_benchmark

    models = args.models or [settings.model]
    agents = args.agents or [settings.agent]
    detectors = args.detectors or [settings.detector]
    rows = []
    for model in models:
        for agent in agents:
            for detector in detectors:
                config = dataclasses.replace(settings, model=model, agent=agent, detector=detector)
                config.validate()
                label = f"{model or 'default'} / {agent} / {detector}"
                _, run_dir = _run_suite(args.suite, config, args, label=label.replace(" / ", "-"))
                rows.append((label, json.loads((run_dir / "report.json").read_text("utf-8")), run_dir))
    table = render_benchmark(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"benchmark-{stamp}.md"
    out.write_text(table, "utf-8")
    print(table)
    print(f"Saved to {out}. Compare any two runs with `carebot eval compare`.")
    return 0


def cmd_eval_compare(args, settings: Settings) -> int:
    from .evals.runner import compare_reports

    print(compare_reports(Path(args.baseline), Path(args.candidate)))
    return 0


def cmd_eval_export_ratings(args, settings: Settings) -> int:
    from .evals.human import export_ratings

    count = export_ratings(args.report, args.out)
    print(f"Wrote {count} conversations to {args.out}.")
    print("Give each rater a copy; they score each criterion 1-5 and put their name in `rater`.")
    print(f"Then run: carebot eval agreement {args.report} <rater1>.csv <rater2>.csv ...")
    return 0


def cmd_eval_agreement(args, settings: Settings) -> int:
    from .evals.human import agreement, render_agreement

    result = agreement(args.report, args.sheets)
    print(json.dumps(result, indent=2) if args.json else render_agreement(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
