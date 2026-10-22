"""A model that plays a caregiver, so the bot can be tested on conversations nobody scripted."""

from __future__ import annotations

SIMULATOR_PROMPT = """\
You are helping test a support chatbot for family caregivers by role-playing a
user. Stay in character for the whole conversation.

Your character:
{persona}

Write only your next text message: one to three short sentences, the way this
person would actually type on their phone. Reveal things gradually, as a real
person would, rather than all at once. Never mention that this is a test or
that you are an AI."""

START = "[The chat has just opened. Send your first message.]"


class SimulatedUser:
    def __init__(self, persona: str, model: str = "claude-sonnet-5-5", client=None):
        import anthropic

        self._system = SIMULATOR_PROMPT.format(persona=persona.strip())
        self._model = model
        self._client = client or anthropic.Anthropic()

    def next_message(self, exchanges: list[tuple[str, str]]) -> str:
        """`exchanges` is [(caregiver_message, bot_reply), ...] so far."""
        # From the simulator's side the roles flip: the bot is its "user".
        messages = [{"role": "user", "content": START}]
        for caregiver, bot in exchanges:
            messages.append({"role": "assistant", "content": caregiver})
            messages.append({"role": "user", "content": bot})
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=2000,
            system=self._system,
            messages=messages,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text").strip()
