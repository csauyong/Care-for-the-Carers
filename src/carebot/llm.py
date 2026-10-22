"""Chat model providers behind one small interface.

- AnthropicChat: Claude via the Anthropic API (default).
- OpenAIChat: any OpenAI-compatible server (OpenAI, Ollama, vLLM, LM Studio...).
- ScriptedChat: canned replies and tool calls, no network. Used by tests and offline evals.

All three support tool use. Tools are given in one neutral shape,
{"name", "description", "input_schema"}, and each provider converts it.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]  # provider-native message dict
Tool = dict[str, Any]  # {"name", "description", "input_schema"}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    message: Message | None = None  # the assistant message to append when continuing a tool loop


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message], tools: list[Tool] | None = None) -> Completion: ...

    def tool_results(self, results: list[tuple[ToolCall, str]]) -> list[Message]: ...


class AnthropicChat:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 4000, client=None):
        import anthropic

        self.name = f"anthropic:{model}"
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    def complete(self, system: str, messages: list[Message], tools: list[Tool] | None = None) -> Completion:
        extra = {"tools": [{**t, "strict": True} for t in tools]} if tools else {}
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
            **extra,
        )
        refused = response.stop_reason == "refusal"
        text = "" if refused else "".join(b.text for b in response.content if b.type == "text")
        calls = () if refused else tuple(
            ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"
        )
        return Completion(
            text=text.strip(),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            tool_calls=calls,
            # Appended unchanged, thinking blocks included, so the tool loop stays valid.
            message={"role": "assistant", "content": response.content},
        )

    def tool_results(self, results: list[tuple[ToolCall, str]]) -> list[Message]:
        # All results for one step go back in a single user message.
        blocks = [{"type": "tool_result", "tool_use_id": call.id, "content": output} for call, output in results]
        return [{"role": "user", "content": blocks}]


class OpenAIChat:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, client=None):
        if client is None:
            try:
                import openai
            except ImportError as e:  # pragma: no cover - depends on install extras
                raise RuntimeError("Install the OpenAI extra: pip install 'care-for-the-carers[openai]'") from e
            # Local servers such as Ollama ignore the key but the client requires one.
            client = openai.OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.name = f"openai:{model}"
        self._client = client
        self._model = model

    def complete(self, system: str, messages: list[Message], tools: list[Tool] | None = None) -> Completion:
        extra = {}
        if tools:
            extra["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools
            ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, *messages],
            **extra,
        )
        choice = response.choices[0]
        usage = response.usage
        calls = tuple(
            ToolCall(c.id, c.function.name, _parse_arguments(c.function.arguments))
            for c in (choice.message.tool_calls or [])
        )
        message: Message = {"role": "assistant", "content": choice.message.content or ""}
        if calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in choice.message.tool_calls
            ]
        return Completion(
            text=(choice.message.content or "").strip(),
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason,
            tool_calls=calls,
            message=message,
        )

    def tool_results(self, results: list[tuple[ToolCall, str]]) -> list[Message]:
        return [{"role": "tool", "tool_call_id": call.id, "content": output} for call, output in results]


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


DEFAULT_SCRIPT = [
    "That sounds really hard. I'm here with you 💛",
    "Thank you for telling me. How are you holding up?",
    "You're doing so much. Have you had a moment for yourself today?",
]


@dataclass
class ScriptedStep:
    """One scripted model response: text, tool calls, or both."""

    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


class ScriptedChat:
    """Deterministic stand-in for a real model. Records every call it gets.

    `replies` items are plain strings or ScriptedSteps (to script tool calls).
    """

    name = "scripted"

    def __init__(self, replies: list[str | ScriptedStep] | None = None, fail: bool = False):
        self._replies = itertools.cycle(replies or DEFAULT_SCRIPT)
        self._fail = fail
        self._ids = itertools.count(1)
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, system: str, messages: list[Message], tools: list[Tool] | None = None) -> Completion:
        self.calls.append((system, [dict(m) for m in messages]))
        if self._fail:
            raise RuntimeError("scripted failure")
        step = next(self._replies)
        if isinstance(step, str):
            step = ScriptedStep(text=step)
        calls = tuple(ToolCall(f"call_{next(self._ids)}", name, args) for name, args in step.tool_calls) if tools else ()
        content: list[dict[str, Any]] = [{"type": "text", "text": step.text}] if step.text else []
        content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
        return Completion(
            text=step.text, model="scripted", tool_calls=calls, message={"role": "assistant", "content": content}
        )

    def tool_results(self, results: list[tuple[ToolCall, str]]) -> list[Message]:
        blocks = [{"type": "tool_result", "tool_use_id": call.id, "content": output} for call, output in results]
        return [{"role": "user", "content": blocks}]
