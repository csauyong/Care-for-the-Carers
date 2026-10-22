"""Retrieval, statistics, human-rating agreement, red-team checks and the benchmark command."""

import csv
import json
from pathlib import Path

import pytest

from carebot.cli import main
from carebot.evals.checks import prompt_leak
from carebot.evals.human import agreement, export_ratings
from carebot.evals.stats import bootstrap_ci, bootstrap_diff_ci, cohen_kappa, spearman, weighted_kappa
from carebot.prompts import SYSTEM_PROMPT
from carebot.resources import Resources
from carebot.retrieval import ResourceRetriever, tokenize

ROOT = Path(__file__).resolve().parents[1]


def test_tokenize_english_and_cantonese():
    assert tokenize("Courses on wandering") == ["course", "wander"]
    assert tokenize("認知障礙") == ["認知", "知障", "障礙"]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("my dad had a stroke and needs a wheelchair", {"stroke-hksr", "stroke-association"}),
        ("阿媽有認知障礙", {"dementia-dcss", "hkada"}),
        ("hello there", {"carer-hotline"}),  # nothing specific: general carer support
    ],
)
def test_retriever_top_pick(query, expected):
    top = ResourceRetriever(Resources.load().support).search(query)
    assert top[0].id in expected


def test_every_bank_entry_is_complete():
    for c in Resources.load().support:
        assert c.id and c.tags and c.keywords and c.source, c.name
        assert c.phone or c.url, c.name


def test_retrieval_eval_gate(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main(["--provider", "scripted", "eval", "retrieval", "--split", "dev", "--min-recall", "0.8"]) == 0
    assert main(["--provider", "scripted", "eval", "retrieval", "--min-recall", "1.01"]) == 1


def test_prompt_leak_detection():
    copied = "Sure! " + " ".join(SYSTEM_PROMPT.split()[:20])
    assert prompt_leak(copied)
    assert not prompt_leak("I'm here for you. How are you holding up today?")


def test_agreement_statistics():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0])) < 1e-9
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    # Off by one everywhere is far better than reversed.
    assert weighted_kappa([1, 2, 3, 4, 5], [2, 3, 4, 5, 5]) > weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    low, high = bootstrap_ci([1.0] * 8 + [0.0] * 2)
    assert low < 0.8 < high
    low, high = bootstrap_diff_ci([0.0] * 10, [1.0] * 10)
    assert low > 0


def _fake_report(tmp_path):
    trials = []
    for i, (empathy, safety) in enumerate([(5, 5), (4, 2), (2, 4), (3, 3)]):
        trials.append({
            "case_id": f"case{i}", "trial": 1, "passed": True, "error": None, "checks": [],
            "transcript": [{"user": "hi", "model_text": "hello", "app_text": "", "trajectory": []}],
            "judge": {"scores": {"empathy": empathy, "safety": safety}, "summary": "", "passed": True},
        })
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"meta": {}, "trials": trials}))
    return path


def test_human_rating_round_trip(tmp_path):
    report = _fake_report(tmp_path)
    sheet = tmp_path / "ratings.csv"
    assert export_ratings(report, sheet) == 4
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert set(rows[0]) >= {"item_id", "conversation", "empathy", "safety", "rater"}
    assert all(r["empathy"] == "" for r in rows)  # blind: the judge's scores aren't in the sheet

    judge = {"case0": (5, 5), "case1": (4, 2), "case2": (2, 4), "case3": (3, 3)}
    for name, shift in (("alice", 0), ("bob", 1)):
        out = tmp_path / f"{name}.csv"
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for r in rows:
                e, s = judge[r["case_id"]]
                writer.writerow({**r, "empathy": min(e + shift, 5), "safety": s, "rater": name})
    result = agreement(report, [tmp_path / "alice.csv", tmp_path / "bob.csv"])
    alice = result["judge_vs_human"]["alice"]["by_criterion"]
    assert alice["empathy"]["exact"] == 1.0 and alice["empathy"]["weighted_kappa"] == 1.0
    bob = result["judge_vs_human"]["bob"]["by_criterion"]["empathy"]
    assert bob["exact"] < 1.0 and bob["within_1"] == 1.0
    assert "alice vs bob" in result["human_vs_human"]


def test_benchmark_and_compare_offline(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(ROOT)
    code = main([
        "--provider", "scripted", "eval", "benchmark", "--agents", "policy", "tools",
        "--only", "safety", "--trials", "2", "--out", str(tmp_path),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "default / policy / keyword" in out and "default / tools / keyword" in out
    runs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(runs) == 2 and list(tmp_path.glob("benchmark-*.md"))
    assert main(["eval", "compare", str(runs[0]), str(runs[1])]) == 0
    assert "OVERALL" in capsys.readouterr().out


def test_redteam_scenarios_pass_offline(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    assert main(["--provider", "scripted", "eval", "scenarios", "--only", "redteam", "--out", str(tmp_path)]) == 0
