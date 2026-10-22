"""Runs the real Anthropic SDK against a local fake API to pin down request shapes."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic
import pytest

from carebot.cli import main
from carebot.evals.judge import DEFAULT_CRITERIA, Judge
from carebot.evals.simulator import SimulatedUser
from carebot.llm import AnthropicChat


class FakeAPI:
    def __init__(self):
        self.requests = []
        self.reply_text = "hello"
        self.stop_reason = "end_turn"


@pytest.fixture
def api():
    state = FakeAPI()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            text = state.reply_text(body) if callable(state.reply_text) else state.reply_text
            payload = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": text}],
                "stop_reason": state.stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{server.server_address[1]}", max_retries=0)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def test_chat_request_shape(api):
    chat = AnthropicChat(model="claude-opus-5-5", effort="low", client=api.client)
    completion = chat.complete("be kind", [{"role": "user", "content": "hi"}])
    assert completion.text == "hello" and completion.input_tokens == 11

    req = api.requests[-1]
    body = req["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "be kind"
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert "temperature" not in body and "thinking" not in body
    assert "server-side-fallback-2026-07-01" in req["headers"]["anthropic-beta"]


def test_refusal_becomes_empty_text(api):
    api.stop_reason = "refusal"
    assert AnthropicChat(client=api.client).complete("s", [{"role": "user", "content": "x"}]).text == ""


def test_judge_parses_structured_verdict(api):
    scores = [{"criterion": name, "score": 4, "reason": "fine"} for name in DEFAULT_CRITERIA]
    scores.append({"criterion": "scenario_1", "score": 2, "reason": "missed it"})
    api.reply_text = json.dumps({"scores": scores, "summary": "mostly good"})

    result = Judge(client=api.client).grade("Caregiver: hi\nBot: hello", ("Mentions the weather",))
    assert result.scores["empathy"] == 4 and result.scores["scenario_1"] == 2
    assert not result.passed  # one criterion below the pass score
    assert "json_schema" in json.dumps(api.requests[-1]["body"]["output_config"])
    assert "Mentions the weather" in api.requests[-1]["body"]["system"]


def test_judge_missing_criterion_fails(api):
    api.reply_text = json.dumps({"scores": [{"criterion": "empathy", "score": 5, "reason": "x"}], "summary": ""})
    result = Judge(client=api.client).grade("Caregiver: hi\nBot: hello")
    assert result.scores["safety"] == 0 and not result.passed


def test_simulator_flips_roles(api):
    api.reply_text = "my back hurts"
    user = SimulatedUser("Mr Chan", client=api.client)
    assert user.next_message([("hi", "hello, how are you?")]) == "my back hurts"
    roles = [m["role"] for m in api.requests[-1]["body"]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_live_eval_commands_end_to_end(api, monkeypatch, tmp_path):
    """`carebot eval personas` and `scenarios --judge` through the real SDK: simulator, bot and judge."""

    def reply(body):
        if "output_config" in body and "format" in body["output_config"]:  # the judge
            return json.dumps({"scores": [{"criterion": c, "score": 4, "reason": "ok"} for c in criteria(body)],
                               "summary": "fine"})
        if "role-playing a\nuser" in body["system"]:  # the simulated caregiver
            return "I'm exhausted and overwhelmed, I can't sleep"
        last = body["messages"][-1]["content"]  # the bot, in the caregiver's language
        return "聽到你咁講，我好擔心你。" if re.search(r"[\u4e00-\u9fff]", last) else "That sounds so hard."

    def criteria(body):
        return [line[2:].split(":")[0] for line in body["system"].splitlines() if line.startswith("- ")]

    api.reply_text = reply
    monkeypatch.setenv("ANTHROPIC_BASE_URL", api.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    out = tmp_path / "personas"
    code = main(["eval", "personas", "--only", "exhausted-daughter", "--out", str(out), "--min-pass-rate", "0"])
    assert code == 0
    report = json.loads(next(out.iterdir()).joinpath("report.json").read_text())
    trial = report["trials"][0]
    assert trial["error"] is None, trial["error"]
    assert len(trial["transcript"]) == 8
    assert trial["judge"]["passed"] and trial["passed"]
    assert "offer_resources" in trial["transcript"][0]["actions"]

    out = tmp_path / "scenarios"
    assert main(["eval", "scenarios", "--judge", "--only", "safety", "--out", str(out)]) == 0
