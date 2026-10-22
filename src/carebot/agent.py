"""The care agent: everything between a user's message and the reply.

Two modes, so they can be compared on the same evals:

- policy: the model only writes text. Code decides when to offer services
  (stress threshold with decay and cooldown).
- tools:  the model is an agent with three tools (search_services,
  offer_services, escalate_crisis) and decides for itself, within guardrails.

In both modes the harness guarantees the safety floor: the crisis screen runs
on every message, and crisis lines are shown whether or not the model calls
escalate_crisis. Every tool call is recorded as a trajectory for review.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .llm import ChatModel, Completion, ToolCall
from .prompts import system_prompt
from .resources import Resources
from .retrieval import ResourceRetriever
from .signals import Detector, Signals
from .store import MemoryStore, Session
from .tracing import Tracer

log = logging.getLogger(__name__)

OFFER_RESOURCES = "offer_resources"
CRISIS_RESOURCES = "crisis_resources"
MODEL_ERROR = "model_error"
MODES = ("policy", "tools")

TOOLS = [
    {
        "name": "search_services",
        "description": "Search the bank of verified Hong Kong support services. Returns up to 3 matches with ids. "
        "Use it before naming any service, phone number or website.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What the carer needs, in a few words."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "offer_services",
        "description": "Offer the carer a list of support services, shown under your reply with Yes/No buttons. "
        "Use when they are struggling or ask for practical help, not on every message. "
        "Pass ids from search_services. The app may decline if services were offered recently.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["service_ids", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_crisis",
        "description": "Show crisis hotlines under your reply. Use whenever there is any sign the carer may harm "
        "themselves or the person they care for, including indirect hints.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class Reply:
    text: str
    buttons: list[tuple[str, str]] = field(default_factory=list)  # (label, callback data)
    actions: list[str] = field(default_factory=list)
    model_text: str = ""  # the part written by the model, before the harness adds anything
    signals: Signals | None = None
    completion: Completion | None = None  # last model response; token counts cover the whole turn
    latency_ms: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    escalation: str | None = None  # "model", "harness" or "both" when crisis lines were shown
    recommended: list[str] = field(default_factory=list)  # service ids shown or offered this turn


@dataclass
class _Turn:
    """What the tools asked for during one turn."""

    offer_ids: list[str] | None = None
    escalated: bool = False


class CareAgent:
    def __init__(
        self,
        model: ChatModel,
        detector: Detector,
        store=None,
        resources: Resources | None = None,
        tracer: Tracer | None = None,
        *,
        mode: str = "policy",
        stress_threshold: float = 3.0,
        stress_decay: float = 0.85,
        offer_cooldown: int = 10,
        history_turns: int = 20,
        max_tool_steps: int = 4,
        retriever=None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.model = model
        self.detector = detector
        self.store = store or MemoryStore()
        self.resources = resources or Resources.load()
        self.retriever = retriever or ResourceRetriever(self.resources.support)
        self.tracer = tracer or Tracer(None)
        self.mode = mode
        self.stress_threshold = stress_threshold
        self.stress_decay = stress_decay
        self.offer_cooldown = offer_cooldown
        self.history_turns = history_turns
        self.max_tool_steps = max_tool_steps
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

    def start(self, user_id: str) -> Reply:
        self.store.delete(user_id)
        return Reply(self.resources.messages["welcome"])

    def forget(self, user_id: str) -> Reply:
        self.store.delete(user_id)
        self.tracer.record({"event": "forget", "user_id": user_id})
        return Reply(self.resources.messages["forgotten"])

    def choose(self, user_id: str, choice: str) -> Reply:
        """Handle the Yes/No answer to an offer (or /resources, which counts as Yes)."""
        accepted = choice == "yes"
        with self._locks[user_id]:
            session = self.store.get(user_id)
            recommended: list[str] = []
            if accepted:
                contacts = self.resources.by_id(session.offer_ids) or self.recommend(session)
                recommended = [c.id for c in contacts]
                text = self.resources.support_block(contacts)
            else:
                text = self.resources.messages["declined"]
            session.offer_ids = []
            session.history.append({"role": "assistant", "content": text})
            self.store.save(session)
        self.tracer.record({"event": "choice", "user_id": user_id, "accepted": accepted, "recommended": recommended})
        return Reply(text, recommended=recommended)

    def recommend(self, session: Session, k: int = 3):
        """Retrieve services for what the carer has said recently."""
        recent = [m["content"] for m in session.history if m["role"] == "user"][-3:]
        return self.retriever.search(" ".join(recent), k)

    def respond(self, user_id: str, text: str) -> Reply:
        with self._locks[user_id]:
            started = time.monotonic()
            session = self.store.get(user_id)
            signals = self.detector.detect(text)
            session.turns += 1
            session.stress = session.stress * self.stress_decay + signals.stress
            session.history.append({"role": "user", "content": text})

            reply = Reply("", signals=signals)
            turn = _Turn()
            try:
                if self.mode == "tools":
                    model_text = self._tool_loop(session, signals, reply, turn)
                else:
                    reply.completion = self.model.complete(
                        system_prompt(self.resources, crisis=signals.crisis), self._window(session)
                    )
                    model_text = reply.completion.text
            except Exception:
                log.exception("model call failed")
                reply.actions.append(MODEL_ERROR)
                model_text = ""
            model_text = model_text or self.resources.messages["error"]
            reply.text = reply.model_text = model_text

            if signals.crisis or turn.escalated:
                reply.text += "\n\n" + self.resources.crisis_block()
                reply.actions.append(CRISIS_RESOURCES)
                reply.escalation = "both" if signals.crisis and turn.escalated else ("model" if turn.escalated else "harness")
            elif turn.offer_ids is not None or (self.mode == "policy" and self._may_offer(session)):
                reply.text += "\n\n" + self.resources.messages["offer"]
                reply.buttons = [("Yes, please", "yes"), ("Not now", "no")]
                reply.actions.append(OFFER_RESOURCES)
                reply.recommended = turn.offer_ids or []
                session.offer_ids = turn.offer_ids or []
                session.last_offer_turn = session.turns
                session.stress = 0.0
            reply.latency_ms = int((time.monotonic() - started) * 1000)

            session.history.append({"role": "assistant", "content": reply.text})
            session.history = session.history[-self.history_turns * 4 :]
            self.store.save(session)
        self._trace(user_id, text, reply, session)
        return reply

    def _tool_loop(self, session: Session, signals: Signals, reply: Reply, turn: _Turn) -> str:
        system = system_prompt(self.resources, crisis=signals.crisis, tools=True)
        messages = self._window(session)
        texts: list[str] = []
        tokens_in = tokens_out = 0
        for _ in range(self.max_tool_steps):
            completion = self.model.complete(system, messages, tools=TOOLS)
            tokens_in += completion.input_tokens
            tokens_out += completion.output_tokens
            if completion.text:
                texts.append(completion.text)
            if not completion.tool_calls:
                break
            results = [(call, self._run_tool(call, session, turn, reply)) for call in completion.tool_calls]
            messages = messages + [completion.message] + self.model.tool_results(results)
        reply.completion = Completion(
            text=completion.text,
            model=completion.model,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            stop_reason=completion.stop_reason,
        )
        return texts[-1] if texts else ""

    def _run_tool(self, call: ToolCall, session: Session, turn: _Turn, reply: Reply) -> str:
        args = call.input
        if call.name == "search_services":
            hits = self.retriever.search(str(args.get("query", "")), k=3)
            result = json.dumps(
                [{"id": c.id, "name": c.name, "phone": c.phone, "url": c.url, "about": c.note} for c in hits],
                ensure_ascii=False,
            )
        elif call.name == "offer_services":
            if not self._cooldown_over(session):
                result = "Not shown: services were offered recently. Keep listening; don't offer again yet."
            else:
                ids = [i for i in args.get("service_ids") or [] if self.resources.by_id([i])]
                turn.offer_ids = ids or [c.id for c in self.recommend(session)]
                result = "Done. Yes/No buttons will appear under your reply. Don't list the services in your text."
        elif call.name == "escalate_crisis":
            turn.escalated = True
            result = "Crisis hotlines will appear under your reply. Don't repeat the numbers; stay with the person."
        else:
            result = f"Unknown tool {call.name!r}."
        reply.trajectory.append({"tool": call.name, "input": args, "result": result})
        return result

    def _may_offer(self, session: Session) -> bool:
        return session.stress >= self.stress_threshold and self._cooldown_over(session)

    def _cooldown_over(self, session: Session) -> bool:
        last = session.last_offer_turn
        return last is None or session.turns - last >= self.offer_cooldown

    def _window(self, session: Session) -> list[dict[str, str]]:
        """Recent history, trimmed so it starts on a user message."""
        window = session.history[-self.history_turns * 2 :]
        while window and window[0]["role"] != "user":
            window = window[1:]
        return window

    def _trace(self, user_id: str, text: str, reply: Reply, session: Session) -> None:
        c = reply.completion
        self.tracer.record(
            {
                "event": "turn",
                "user_id": user_id,
                "turn": session.turns,
                "mode": self.mode,
                "user_text": text,
                "model_text": reply.model_text,
                "reply_text": reply.text,
                "actions": reply.actions,
                "trajectory": reply.trajectory,
                "escalation": reply.escalation,
                "recommended": reply.recommended,
                "detector": self.detector.name,
                "stress_delta": reply.signals.stress if reply.signals else None,
                "stress_total": round(session.stress, 2),
                "crisis": bool(reply.signals and reply.signals.crisis),
                "evidence": reply.signals.evidence if reply.signals else [],
                "model": c.model if c else self.model.name,
                "stop_reason": c.stop_reason if c else None,
                "input_tokens": c.input_tokens if c else 0,
                "output_tokens": c.output_tokens if c else 0,
                "latency_ms": reply.latency_ms,
            }
        )
