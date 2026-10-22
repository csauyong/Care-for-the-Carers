"""Locale data: support contacts and the bot's fixed messages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from urllib.parse import urlparse

import yaml

_PHONE_RE = re.compile(r"(?<![\d.])(?:\+?852[\s-]?)?(?:\d{3,4}[\s-]?\d{3,4}|999)(?![\d.])")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)>\]\"']+", re.IGNORECASE)


@dataclass(frozen=True)
class Contact:
    name: str
    phone: str | None = None
    url: str | None = None
    note: str | None = None
    id: str = ""
    tags: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    source: str | None = None

    def render(self) -> str:
        detail = " · ".join(x for x in (self.phone, self.url) if x)
        line = f"• {self.name}: {detail}"
        return f"{line}\n  {self.note}" if self.note else line


@dataclass(frozen=True)
class Resources:
    locale: str
    crisis_lines: tuple[Contact, ...]
    support: tuple[Contact, ...]
    messages: dict[str, str]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Resources":
        if path is None:
            text = importlib_resources.files("carebot.data").joinpath("hong_kong.yaml").read_text("utf-8")
        else:
            text = Path(path).read_text("utf-8")
        raw = yaml.safe_load(text)
        return cls(
            locale=raw["locale"],
            crisis_lines=tuple(_contact(c) for c in raw["crisis_lines"]),
            support=tuple(_contact(c) for c in raw["support"]),
            messages=raw["messages"],
        )

    def crisis_block(self) -> str:
        lines = "\n".join(c.render() for c in self.crisis_lines)
        return f"{self.messages['crisis']}\n{lines}"

    def support_block(self, contacts: list[Contact] | None = None) -> str:
        lines = "\n".join(c.render() for c in (contacts or self.support))
        return f"{self.messages['accepted']}\n{lines}\n\n{self.messages['accepted_outro']}"

    def by_id(self, ids: list[str]) -> list[Contact]:
        index = {c.id: c for c in self.support}
        return [index[i] for i in ids if i in index]

    def for_prompt(self) -> str:
        """Plain list of every contact, for the system prompt."""
        return "\n".join(c.render() for c in (*self.crisis_lines, *self.support))

    def unlisted_contacts(self, text: str) -> list[str]:
        """Phone numbers or links in `text` that are not in this file."""
        contacts = (*self.crisis_lines, *self.support)
        phones = {_digits(c.phone) for c in contacts if c.phone}
        hosts = {_host(c.url) for c in contacts if c.url}
        found = [m.group(0) for m in _PHONE_RE.finditer(text) if _digits(m.group(0)) not in phones]
        found += [m.group(0) for m in _URL_RE.finditer(text) if _host(m.group(0)) not in hosts]
        return found


def _contact(raw: dict) -> Contact:
    raw = dict(raw)
    for key in ("tags", "keywords"):
        raw[key] = tuple(raw.get(key) or ())
    return Contact(**raw)


def _digits(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    return digits[3:] if digits.startswith("852") and len(digits) > 8 else digits


def _host(url: str) -> str:
    if not url.lower().startswith("http"):
        url = "https://" + url
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host
