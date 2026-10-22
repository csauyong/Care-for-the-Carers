"""Eval case files (YAML) and their loaders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Turn:
    """A user message, or a tap on the Yes/No offer buttons (`choose: yes`)."""

    user: str | None = None
    choose: str | None = None
    expect: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    """A scripted conversation: fixed user messages, expectations per turn."""

    id: str
    description: str
    turns: tuple[Turn, ...]
    expect: dict[str, Any] = field(default_factory=dict)
    rubric: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Persona:
    """A simulated caregiver: a model plays the user for `turns` messages."""

    id: str
    description: str
    persona: str
    turns: int = 6
    opening: str | None = None
    accepts_offers: bool = True  # taps "Yes" when the bot offers services
    expect: dict[str, Any] = field(default_factory=dict)
    rubric: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecommendationCase:
    """What a carer said, and which services in the resource bank would actually help."""

    id: str
    messages: tuple[str, ...]
    relevant: tuple[str, ...]
    split: str = "test"  # tune on "dev"; report "test"


@dataclass(frozen=True)
class DetectorCase:
    text: str
    stressed: bool
    crisis: bool = False
    note: str = ""


def load_scenarios(directory: str | Path, only: list[str] | None = None) -> list[Scenario]:
    scenarios = []
    for raw in _load_dir(directory):
        turns = tuple(Turn(t.get("user"), t.get("choose"), t.get("expect") or {}) for t in raw["turns"])
        scenarios.append(
            Scenario(
                id=raw["id"],
                description=raw.get("description", ""),
                turns=turns,
                expect=raw.get("expect") or {},
                rubric=tuple(raw.get("rubric") or ()),
                tags=tuple(raw.get("tags") or ()),
            )
        )
    return _filter(scenarios, only)


def load_personas(directory: str | Path, only: list[str] | None = None) -> list[Persona]:
    personas = [
        Persona(
            id=raw["id"],
            description=raw.get("description", ""),
            persona=raw["persona"],
            turns=int(raw.get("turns", 6)),
            opening=raw.get("opening"),
            accepts_offers=bool(raw.get("accepts_offers", True)),
            expect=raw.get("expect") or {},
            rubric=tuple(raw.get("rubric") or ()),
            tags=tuple(raw.get("tags") or ()),
        )
        for raw in _load_dir(directory)
    ]
    return _filter(personas, only)


def load_detector_cases(path: str | Path) -> list[DetectorCase]:
    raw = yaml.safe_load(Path(path).read_text("utf-8"))
    return [DetectorCase(**case) for case in raw["cases"]]


def load_recommendation_cases(path: str | Path, split: str = "all") -> list[RecommendationCase]:
    raw = yaml.safe_load(Path(path).read_text("utf-8"))
    cases = [
        RecommendationCase(c["id"], tuple(c["messages"]), tuple(c["relevant"]), c.get("split", "test"))
        for c in raw["cases"]
    ]
    return [c for c in cases if split == "all" or c.split == split]


def dataset_version(*paths: str | Path) -> str:
    """Hash of every eval file used, so reports show when the test set itself changed."""
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
        for f in files:
            digest.update(f.name.encode() + f.read_bytes())
    return digest.hexdigest()[:10]


def _load_dir(directory: str | Path) -> list[dict[str, Any]]:
    paths = sorted(Path(directory).glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No .yaml eval files in {directory}")
    return [yaml.safe_load(p.read_text("utf-8")) for p in paths]


def _filter(cases, only):
    if not only:
        return cases
    picked = [c for c in cases if c.id in only or set(c.tags) & set(only)]
    if not picked:
        raise ValueError(f"No eval cases match {only}")
    return picked
