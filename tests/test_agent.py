from carebot.agent import CRISIS_RESOURCES, MODEL_ERROR, OFFER_RESOURCES, CareAgent
from carebot.llm import ScriptedChat
from carebot.signals import KeywordDetector
from carebot.store import MemoryStore
from carebot.tracing import Tracer


def make_agent(model=None, **kwargs):
    return CareAgent(model or ScriptedChat(["ok"]), KeywordDetector(), MemoryStore(), **kwargs)


def test_calm_message_gets_model_reply_only():
    agent = make_agent()
    reply = agent.respond("u1", "Mum had a good day today")
    assert reply.text == "ok"
    assert reply.actions == []
    assert reply.buttons == []


def test_offers_resources_once_when_stress_builds():
    agent = make_agent()
    assert agent.respond("u1", "I'm exhausted").actions == []
    reply = agent.respond("u1", "I'm overwhelmed and anxious, I can't sleep")
    assert reply.actions == [OFFER_RESOURCES]
    assert [data for _, data in reply.buttons] == ["yes", "no"]
    # Still stressed on the next message, but the offer is on cooldown.
    assert agent.respond("u1", "still stressed and upset and anxious").actions == []


def test_offer_returns_after_cooldown():
    agent = make_agent(offer_cooldown=2)
    agent.respond("u1", "overwhelmed anxious exhausted")
    agent.respond("u1", "fine")
    assert OFFER_RESOURCES in agent.respond("u1", "overwhelmed anxious exhausted").actions


def test_crisis_appends_hotlines_and_tells_the_model():
    model = ScriptedChat(["I'm so sorry you're feeling this way."])
    agent = make_agent(model)
    reply = agent.respond("u1", "I don't want to wake up anymore")
    assert reply.actions == [CRISIS_RESOURCES]
    assert "18111" in reply.text and "2896 0000" in reply.text
    system, _ = model.calls[-1]
    assert "risk of harm" in system


def test_crisis_lines_still_shown_when_model_fails():
    agent = make_agent(ScriptedChat(fail=True))
    reply = agent.respond("u1", "我真係好想死")
    assert MODEL_ERROR in reply.actions and CRISIS_RESOURCES in reply.actions
    assert "18111" in reply.text


def test_users_are_isolated():
    agent = make_agent()
    agent.respond("a", "overwhelmed anxious")
    assert agent.respond("b", "exhausted").actions == []


def test_history_window_starts_with_user_and_is_bounded():
    model = ScriptedChat(["ok"])
    agent = make_agent(model, history_turns=2)
    agent.choose("u1", "no")  # assistant message before any user message
    for i in range(5):
        agent.respond("u1", f"message {i}")
    _, messages = model.calls[-1]
    assert messages[0]["role"] == "user"
    assert len(messages) <= 4
    assert messages[-1]["content"] == "message 4"


def test_choose_yes_lists_support_services():
    agent = make_agent()
    assert "182 183" in agent.choose("u1", "yes").text
    assert "182 183" not in agent.choose("u1", "no").text


def test_start_and_forget_clear_state():
    agent = make_agent()
    agent.respond("u1", "overwhelmed anxious")
    agent.forget("u1")
    assert agent.store.get("u1").history == []


def test_traces_hash_user_and_can_drop_content(tmp_path):
    path = tmp_path / "traces.jsonl"
    agent = make_agent(tracer=Tracer(path, include_content=False))
    agent.respond("chat-777", "I'm exhausted")
    line = path.read_text()
    assert "chat-777" not in line and "exhausted" not in line
    assert '"turn": 1' in line
