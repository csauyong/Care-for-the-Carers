"""Builds an agent from Settings, so every entry point wires things the same way."""

from __future__ import annotations

from .agent import CareAgent
from .config import Settings
from .llm import AnthropicChat, ChatModel, OpenAIChat, ScriptedChat
from .resources import Resources
from .retrieval import LLMRecommender, ResourceRetriever
from .signals import Detector, FinetunedDetector, KeywordDetector, LLMDetector
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
        return LLMDetector(model=_claude_model(settings))
    if settings.detector == "finetuned":
        return FinetunedDetector(settings.finetuned_model)
    return KeywordDetector()


def _claude_model(settings: Settings) -> str:
    """Claude-only helpers (LLM detector, LLM recommender) use Claude whatever provider writes replies."""
    return settings.model if settings.provider == "anthropic" and settings.model else DEFAULT_ANTHROPIC_MODEL


def build_retriever(settings: Settings, resources: Resources):
    if settings.retriever == "llm":
        return LLMRecommender(resources.support, model=_claude_model(settings))
    return ResourceRetriever(resources.support)


def build_agent(settings: Settings, *, persistent: bool = True, model: ChatModel | None = None) -> CareAgent:
    """`persistent=False` keeps sessions in memory and skips traces (used by evals)."""
    store = SQLiteStore(settings.data_dir / "sessions.db") if persistent else MemoryStore()
    tracer = Tracer(settings.data_dir / "traces.jsonl" if persistent else None, settings.trace_content)
    resources = Resources.load(settings.resources_file)
    return CareAgent(
        model=model or build_model(settings),
        detector=build_detector(settings),
        store=store,
        resources=resources,
        retriever=build_retriever(settings, resources),
        tracer=tracer,
        mode=settings.agent,
        stress_threshold=settings.stress_threshold,
        history_turns=settings.history_turns,
    )
