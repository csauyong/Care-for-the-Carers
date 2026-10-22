"""Append-only JSONL traces of every turn, for debugging and offline evals.

User ids are hashed. Set CAREBOT_TRACE_CONTENT=false to drop message text
and keep only the signals, actions, latency and token counts.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

_CONTENT_KEYS = ("user_text", "reply_text", "model_text", "evidence")


class Tracer:
    def __init__(self, path: str | Path | None, include_content: bool = True):
        self.path = Path(path) if path else None
        self.include_content = include_content
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict[str, Any]) -> None:
        if not self.path:
            return
        event = {"ts": time.time(), **event}
        if "user_id" in event:
            event["user_id"] = pseudonymize(event["user_id"])
        if not self.include_content:
            event = {k: v for k, v in event.items() if k not in _CONTENT_KEYS}
        line = json.dumps(event, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def pseudonymize(user_id: str) -> str:
    return hashlib.sha256(f"carebot:{user_id}".encode()).hexdigest()[:12]
