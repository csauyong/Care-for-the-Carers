"""The care agent: everything between a user's message and the reply.

The model writes the conversational text. The harness around it decides,
deterministically and testably, when to escalate to crisis lines, when to offer
support services, what history the model sees, and what gets traced.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .llm import ChatModel, Completion
from .prompts import system_prompt
from .resources import Resources
from .signals import Detector, Signals
from .store import MemoryStore, Session
from .tracing import Tracer

log = logging.getLogger(__name__)

OFFER_RESOURCES = "offer_resources"
CRISIS_RESOURCES = "crisis_resources"
MODEL_ERROR = "model_error"


@dataclass
class Reply:
    text: str
    buttons: list[tuple[str, str]] = field(default_factory=list)  # (label, callback data)
    actions: list[str] = field(default_factory=list)
    model_text: str = ""  # the part written by the model, before the harness adds anything
    signals: Signals | None = None
    completion: Completion | None = None
    latency_ms: int = 0


class CareAgent:
    def __init__(
        self,
        model: ChatModel,
        detector: Detector,
        store=None,
        resources: Resources | None = None,
        tracer: Tracer | None = None,
        *,
        stress_threshold: float = 3.0,
        stress_decay: float = 0.85,
        offer_cooldown: int = 10,
        history_turns: int = 20,
    ):
        self.model = model
        self.detector = detector
        self.store = store or MemoryStore()
        self.resources = resources or Resources.load()
        self.tracer = tracer or Tracer(None)
        self.stress_threshold = stress_threshold
        self.stress_decay = stress_decay
        self.offer_cooldown = offer_cooldown
        self.history_turns = history_turns
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

    def start(self, user_id: str) -> Reply:
        self.store.delete(user_id)
        return Reply(self.resources.messages["welcome"])

    def forget(self, user_id: str) -> Reply:
        self.store.delete(user_id)
        self.tracer.record({"event": "forget", "user_id": user_id})
        return Reply(self.resources.messages["forgotten"])

    def choose(self, user_id: str, choice: str) -> Reply:
        """Handle the Yes/No answer to a resource offer."""
        accepted = choice == "yes"
        text = self.resources.support_block() if accepted else self.resources.messages["declined"]
        with self._locks[user_id]:
            session = self.store.get(user_id)
            session.history.append({"role": "assistant", "content": text})
            self.store.save(session)
        self.tracer.record({"event": "choice", "user_id": user_id, "accepted": accepted})
        return Reply(text)

    def respond(self, user_id: str, text: str) -> Reply:
        with self._locks[user_id]:
            started = time.monotonic()
            session = self.store.get(user_id)
            signals = self.detector.detect(text)
            session.turns += 1
            session.stress = session.stress * self.stress_decay + signals.stress
            session.history.append({"role": "user", "content": text})

            actions: list[str] = []
            completion = None
            try:
                completion = self.model.complete(
                    system_prompt(self.resources, crisis=signals.crisis), self._window(session)
                )
                model_text = completion.text
            except Exception:
                log.exception("model call failed")
                actions.append(MODEL_ERROR)
                model_text = ""
            if not model_text:
                model_text = self.resources.messages["error"]

            reply = Reply(model_text, model_text=model_text, signals=signals, completion=completion)
            if signals.crisis:
                reply.text += "\n\n" + self.resources.crisis_block()
                actions.append(CRISIS_RESOURCES)
            elif self._should_offer(session):
                reply.text += "\n\n" + self.resources.messages["offer"]
                reply.buttons = [("Yes, please", "yes"), ("Not now", "no")]
                actions.append(OFFER_RESOURCES)
                session.last_offer_turn = session.turns
                session.stress = 0.0
            reply.actions = actions
            reply.latency_ms = int((time.monotonic() - started) * 1000)

            session.history.append({"role": "assistant", "content": reply.text})
            session.history = session.history[-self.history_turns * 4 :]
            self.store.save(session)
        self._trace(user_id, text, reply, session)
        return reply

    def _should_offer(self, session: Session) -> bool:
        if session.stress < self.stress_threshold:
            return False
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
                "user_text": text,
                "model_text": reply.model_text,
                "reply_text": reply.text,
                "actions": reply.actions,
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
