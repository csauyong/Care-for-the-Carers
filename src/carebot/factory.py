"""Builds an agent from Settings, so every entry point wires things the same way."""

from __future__ import annotations

from .agent import CareAgent
from .config import Settings
from .llm import AnthropicChat, ChatModel, OpenAIChat, ScriptedChat
from .resources import Resources
from .signals import Detector, KeywordDetector, LLMDetector
from .store import MemoryStore, SQLiteStore
from .tracing import Tracer

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


def build_model(settings: Settings) -> ChatModel:
    if settings.provider == "anthropic":
        return AnthropicChat(model=settings.model or DEFAULT_ANTHROPIC_MODEL, effort=settings.effort)
    if settings.provider == "openai":
        return OpenAIChat(model=settings.model, base_url=settings.openai_base_url)
    return ScriptedChat()


def build_detector(settings: Settings) -> Detector:
    if settings.detector == "llm":
        # The LLM detector always uses Claude, whatever provider writes the replies.
        model = settings.model if settings.provider == "anthropic" and settings.model else DEFAULT_ANTHROPIC_MODEL
        return LLMDetector(model=model)
    return KeywordDetector()


def build_agent(settings: Settings, *, persistent: bool = True, model: ChatModel | None = None) -> CareAgent:
    """`persistent=False` keeps sessions in memory and skips traces (used by evals)."""
    store = SQLiteStore(settings.data_dir / "sessions.db") if persistent else MemoryStore()
    tracer = Tracer(settings.data_dir / "traces.jsonl" if persistent else None, settings.trace_content)
    return CareAgent(
        model=model or build_model(settings),
        detector=build_detector(settings),
        store=store,
        resources=Resources.load(settings.resources_file),
        tracer=tracer,
        stress_threshold=settings.stress_threshold,
        history_turns=settings.history_turns,
    )
