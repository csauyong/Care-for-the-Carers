"""Per-user session state, in memory or in a SQLite file that survives restarts."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Session:
    user_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    stress: float = 0.0
    turns: int = 0
    last_offer_turn: int | None = None


class MemoryStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get(self, user_id: str) -> Session:
        return self._sessions.get(user_id) or Session(user_id)

    def save(self, session: Session) -> None:
        self._sessions[session.user_id] = session

    def delete(self, user_id: str) -> None:
        self._sessions.pop(user_id, None)


class SQLiteStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions (user_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL)"
            )

    def get(self, user_id: str) -> Session:
        with self._lock:
            row = self._conn.execute("SELECT data FROM sessions WHERE user_id = ?", (user_id,)).fetchone()
        return Session(**json.loads(row[0])) if row else Session(user_id)

    def save(self, session: Session) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions (user_id, data, updated_at) VALUES (?, ?, ?)",
                (session.user_id, json.dumps(asdict(session), ensure_ascii=False), time.time()),
            )

    def delete(self, user_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
