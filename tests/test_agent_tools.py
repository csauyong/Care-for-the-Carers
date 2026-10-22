"""tools mode: the model decides, the harness keeps the guarantees."""

from types import SimpleNamespace

from carebot.agent import CRISIS_RESOURCES, OFFER_RESOURCES, CareAgent
from carebot.llm import OpenAIChat, ScriptedChat, ScriptedStep
from carebot.signals import KeywordDetector


def tools_agent(*steps, **kwargs):
    return CareAgent(ScriptedChat(list(steps)), KeywordDetector(), mode="tools", **kwargs)


def test_search_then_offer_sets_buttons_and_recommendations():
    agent = tools_agent(
        ScriptedStep(tool_calls=[("search_services", {"query": "respite"})]),
        ScriptedStep(tool_calls=[("offer_services", {"service_ids": ["respite-elderly", "nope"], "reason": "needs a break"})]),
        ScriptedStep(text="You deserve a break. Want me to share some options?"),
    )
    reply = agent.respond("u", "I have nobody to look after mum while I'm in hospital")
    assert [s["tool"] for s in reply.trajectory] == ["search_services", "offer_services"]
    assert "respite-elderly" in reply.trajectory[0]["result"]
    assert reply.actions == [OFFER_RESOURCES] and reply.recommended == ["respite-elderly"]  # unknown id dropped
    assert reply.model_text == "You deserve a break. Want me to share some options?"
    chosen = agent.choose("u", "yes")
    assert chosen.recommended == ["respite-elderly"] and "ves.swd.gov.hk" in chosen.text


def test_offer_is_declined_during_cooldown():
    offer = ScriptedStep(tool_calls=[("offer_services", {"service_ids": ["carer-hotline"], "reason": "x"})])
    agent = tools_agent(offer, ScriptedStep(text="ok"), offer, ScriptedStep(text="ok"))
    assert OFFER_RESOURCES in agent.respond("u", "hard day").actions
    second = agent.respond("u", "another hard day")
    assert OFFER_RESOURCES not in second.actions
    assert "offered recently" in second.trajectory[0]["result"]


def test_harness_escalates_when_model_does_not():
    agent = tools_agent(ScriptedStep(text="That sounds so heavy."))
    reply = agent.respond("u", "I want to die")
    assert reply.escalation == "harness" and CRISIS_RESOURCES in reply.actions
    assert "18111" in reply.text


def test_model_escalation_without_keyword_match():
    agent = tools_agent(
        ScriptedStep(tool_calls=[("escalate_crisis", {"reason": "indirect hopelessness"})]),
        ScriptedStep(text="I'm here with you."),
    )
    reply = agent.respond("u", "I can't see the point of anything anymore")
    assert reply.escalation == "model" and "18111" in reply.text


def test_no_offer_buttons_on_a_crisis_turn():
    agent = tools_agent(
        ScriptedStep(tool_calls=[("escalate_crisis", {"reason": "r"}), ("offer_services", {"service_ids": [], "reason": "r"})]),
        ScriptedStep(text="I'm here."),
    )
    reply = agent.respond("u", "I want to die")
    assert reply.escalation == "both" and reply.buttons == []


def test_tool_loop_is_bounded():
    agent = tools_agent(ScriptedStep(tool_calls=[("search_services", {"query": "x"})]), max_tool_steps=3)
    reply = agent.respond("u", "hello")
    assert len(reply.trajectory) == 3
    assert reply.model_text  # falls back to the error message, never empty


def test_choose_yes_in_policy_mode_retrieves_for_the_conversation():
    agent = CareAgent(ScriptedChat(["ok"]), KeywordDetector())
    agent.respond("u", "Since his stroke I can't lift him into the wheelchair")
    reply = agent.choose("u", "yes")
    assert reply.recommended[0].startswith("stroke-")


def test_openai_tool_calls_round_trip():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="search_services", arguments='{"query": "money"}'))
        message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            model="llama3.1",
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    chat = OpenAIChat("llama3.1", client=client)
    out = chat.complete("sys", [{"role": "user", "content": "hi"}], tools=[
        {"name": "search_services", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    ])
    assert out.tool_calls[0].input == {"query": "money"}
    assert calls[0]["tools"][0]["function"]["name"] == "search_services"
    assert out.message["tool_calls"][0]["id"] == "c1"
    assert chat.tool_results([(out.tool_calls[0], "[]")]) == [{"role": "tool", "tool_call_id": "c1", "content": "[]"}]
