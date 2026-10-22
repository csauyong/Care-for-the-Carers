"""`carebot` command line.

    carebot run                  start the Telegram bot
    carebot chat                 talk to the bot in your terminal
    carebot check                validate configuration and make one test call
    carebot eval detector        score the stress/crisis detector on labelled messages
    carebot eval scenarios       replay scripted conversations and check the replies
    carebot eval personas        let simulated caregivers talk to the bot, then judge it
    carebot eval compare A B     compare two eval reports
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
    parser.add_argument("--detector", choices=["keyword", "llm"], help="override CAREBOT_DETECTOR")
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

    for name, default_dir, helptext in (
        ("scenarios", "evals/scenarios", "replay scripted conversations"),
        ("personas", "evals/personas", "simulated caregivers (needs Anthropic API access)"),
    ):
        p = ev.add_parser(name, help=helptext)
        p.add_argument("--dir", default=default_dir)
        p.add_argument("--only", nargs="*", help="case ids or tags to run")
        p.add_argument("--trials", type=int, default=1, help="runs per case; models are not deterministic")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--out", default="eval-results")
        p.add_argument("--min-pass-rate", type=float, default=1.0, help="exit non-zero below this (default 1.0)")
        if name == "scenarios":
            p.add_argument("--judge", action="store_true", help="also grade conversations with an LLM judge")
        else:
            p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
        p.set_defaults(func=partial(cmd_eval_suite, name))

    p = ev.add_parser("compare", help="compare two reports (report.json or its folder)")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.set_defaults(func=cmd_eval_compare)
    return parser


def _settings(args) -> Settings:
    settings = Settings.from_env(args.env_file)
    overrides = {k: v for k, v in (("provider", args.provider), ("model", args.model), ("detector", args.detector)) if v}
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
        tags = ", ".join(reply.actions + (reply.signals.evidence if reply.signals else []))
        print(f"     ({reply.latency_ms} ms{'; ' + tags if tags else ''})\n")


def cmd_check(args, settings: Settings) -> int:
    from .factory import build_agent

    print(f"provider={settings.provider} model={settings.model or 'default'} detector={settings.detector}")
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


def cmd_eval_suite(kind: str, args, settings: Settings) -> int:
    from .evals.datasets import load_personas, load_scenarios
    from .evals.judge import Judge
    from .evals.runner import run_persona, run_scenario, run_suite, write_report
    from .factory import build_agent

    use_judge = args.judge if kind == "scenarios" else not args.no_judge
    if kind == "personas" and settings.provider == "scripted":
        raise ConfigError("Persona evals need a real model; use --provider anthropic or openai.")
    judge = Judge(settings.judge_model) if use_judge else None
    make_agent = partial(build_agent, settings, persistent=False)
    if kind == "scenarios":
        cases = load_scenarios(args.dir, args.only)
        model_checks = settings.provider != "scripted"  # canned replies can't meet wording checks
        jobs = [
            partial(run_scenario, make_agent, c, t, judge, model_checks)
            for c in cases
            for t in range(1, args.trials + 1)
        ]
    else:
        cases = load_personas(args.dir, args.only)
        jobs = [
            partial(run_persona, make_agent, c, t, judge, settings.simulator_model)
            for c in cases
            for t in range(1, args.trials + 1)
        ]
    print(f"Running {len(jobs)} {kind} trials ({len(cases)} cases × {args.trials})...")
    results = run_suite(jobs, args.workers)
    meta = {
        "suite": kind,
        "model": make_agent().model.name,
        "detector": settings.detector,
        "judge": judge.model if judge else None,
        "simulator": settings.simulator_model if kind == "personas" else None,
        "trials_per_case": args.trials,
    }
    run_dir = write_report(results, meta, args.out)
    print((run_dir / "report.md").read_text("utf-8"))
    print(f"Full report: {run_dir / 'report.json'}")
    pass_rate = sum(r.passed for r in results) / len(results)
    return 0 if pass_rate >= args.min_pass_rate else 1


def cmd_eval_compare(args, settings: Settings) -> int:
    from .evals.runner import compare_reports

    print(compare_reports(Path(args.baseline), Path(args.candidate)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
