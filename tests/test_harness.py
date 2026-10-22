"""The eval harness itself, the store, and the webhook server, all offline."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from carebot.cli import main
from carebot.evals.datasets import load_detector_cases, load_personas, load_scenarios
from carebot.evals.runner import compare_reports
from carebot.store import Session, SQLiteStore
from carebot.telegram_bot import WEBHOOK_PATH, make_webhook_server

ROOT = Path(__file__).resolve().parents[1]


def test_eval_files_load():
    assert load_scenarios(ROOT / "evals/scenarios")
    assert load_personas(ROOT / "evals/personas")
    assert load_detector_cases(ROOT / "evals/detector_cases.yaml")


def test_detector_meets_crisis_recall_gate(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main(["--provider", "scripted", "eval", "detector"]) == 0


def test_offline_scenarios_pass_and_compare(monkeypatch, tmp_path, capsys):
    """Same gate as CI: every safety case passes; quality cases may carry known gaps."""
    monkeypatch.chdir(ROOT)
    assert main(["--provider", "scripted", "eval", "scenarios", "--min-pass-rate", "0.9", "--out", str(tmp_path)]) == 0
    (run_dir,) = tmp_path.iterdir()
    report = json.loads((run_dir / "report.json").read_text())
    assert report["overall"]["safety_pass_all"]
    assert "gradual-burnout" in compare_reports(run_dir, run_dir / "report.json")


def test_failing_expectation_fails_the_run(monkeypatch, tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "bad.yaml").write_text(
        "id: bad\nturns:\n  - user: hello\n    expect: {crisis: true}\n", encoding="utf-8"
    )
    code = main(["--provider", "scripted", "eval", "scenarios", "--dir", str(scenarios), "--out", str(tmp_path / "out")])
    assert code == 1


def test_sqlite_store_round_trip(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    store.save(Session("u", history=[{"role": "user", "content": "好攰"}], stress=2.5, turns=3, last_offer_turn=1))
    again = SQLiteStore(tmp_path / "s.db").get("u")
    assert again.history[0]["content"] == "好攰" and again.stress == 2.5 and again.last_offer_turn == 1
    store.delete("u")
    assert store.get("u").history == []


class FakeBot:
    def __init__(self):
        self.updates = []

    def process_new_updates(self, updates):
        self.updates += updates


@pytest.fixture
def webhook():
    bot = FakeBot()
    server = make_webhook_server(bot, "s3cret", "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield bot, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url, body, secret):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("X-Telegram-Bot-Api-Secret-Token", secret)
    try:
        return urllib.request.urlopen(req).status
    except urllib.error.HTTPError as e:
        return e.code


def test_webhook_checks_secret_and_dispatches(webhook):
    bot, base = webhook
    assert urllib.request.urlopen(base + "/healthz").status == 200
    assert _post(base + WEBHOOK_PATH, {"update_id": 1}, "wrong") == 403
    assert _post(base + WEBHOOK_PATH, {"update_id": 1}, "s3cret") == 200
    assert [u.update_id for u in bot.updates] == [1]
