"""Chat model providers behind one small interface.

- AnthropicChat: Claude via the Anthropic API (default).
- OpenAIChat: any OpenAI-compatible server (OpenAI, Ollama, vLLM, LM Studio...).
- ScriptedChat: canned replies, no network. Used by tests and offline evals.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol

Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message]) -> Completion: ...


class AnthropicChat:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 4000, client=None):
        import anthropic

        self.name = f"anthropic:{model}"
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    def complete(self, system: str, messages: list[Message]) -> Completion:
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self._effort},
            # If a safety classifier declines, retry server-side on Anthropic's
            # recommended fallback model instead of returning nothing.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        text = "" if response.stop_reason == "refusal" else "".join(
            block.text for block in response.content if block.type == "text"
        )
        return Completion(
            text=text.strip(),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


class OpenAIChat:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        try:
            import openai
        except ImportError as e:  # pragma: no cover - depends on install extras
            raise RuntimeError("Install the OpenAI extra: pip install 'care-for-the-carers[openai]'") from e

        self.name = f"openai:{model}"
        # Local servers such as Ollama ignore the key but the client requires one.
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._model = model

    def complete(self, system: str, messages: list[Message]) -> Completion:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, *messages],
        )
        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=(choice.message.content or "").strip(),
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason,
        )


DEFAULT_SCRIPT = [
    "That sounds really hard. I'm here with you 💛",
    "Thank you for telling me. How are you holding up?",
    "You're doing so much. Have you had a moment for yourself today?",
]


class ScriptedChat:
    """Deterministic stand-in for a real model. Records every call it gets."""

    name = "scripted"

    def __init__(self, replies: list[str] | None = None, fail: bool = False):
        self._replies = itertools.cycle(replies or DEFAULT_SCRIPT)
        self._fail = fail
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, system: str, messages: list[Message]) -> Completion:
        self.calls.append((system, [dict(m) for m in messages]))
        if self._fail:
            raise RuntimeError("scripted failure")
        return Completion(text=next(self._replies), model="scripted")
