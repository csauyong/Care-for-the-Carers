"""Detectors that read a single user message for stress and crisis signals.

The agent never relies on the model to notice a crisis: the keyword crisis
screen always runs, and its output is OR-ed with any LLM detector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

STRESS_TERMS = [
    "overwhelmed", "anxious", "stressed", "stressful", "tense", "exhausted", "worried",
    "nervous", "restless", "irritable", "burned out", "burnt out", "burnout", "fatigued",
    "frustrated", "upset", "distressed", "hopeless", "helpless", "desperate",
    "trapped", "panicked", "panicking", "strained", "agitated", "on edge",
    "difficulty concentrating", "can't focus", "can't sleep", "cannot sleep", "losing sleep",
    "sleepless", "insomnia", "racing thoughts", "anxiety", "depressed", "depression",
    "crying", "mood swings", "can't relax", "feeling low", "lack of motivation",
    "unmotivated", "overthinking", "can't cope", "cannot cope", "breaking point",
    "at the end of my rope", "falling apart", "no time for myself",
    # Cantonese / Traditional Chinese
    "好攰", "好癐", "壓力", "好煩", "頂唔順", "瞓唔着", "瞓唔著", "失眠", "焦慮", "抑鬱", "崩潰", "好辛苦",
]

# Deliberately broad: a false alarm costs one extra message, a miss can cost a life.
CRISIS_PATTERNS = [
    r"\bsuicid",
    r"\bkill(ing)? myself\b",
    r"\b(hurt|harm|cut)(ing)? myself\b",
    r"\bend(ing)? (it all|my life|everything)\b",
    r"\b(want|wanna|wish|going) (to )?die\b",
    r"\bwish i (was|were) dead\b",
    r"\b(don'?t|do not) want to (live|be alive|be here|wake up)( anymore)?\b",
    r"\bno (reason|point) (to|in) (live|living|going on)\b",
    r"\b(better off|be better) without me\b",
    r"\boverdose\b",
    r"\b(take|taking|took|swallow|swallowing) (all|a lot of|a bottle of) (my|the|her|his|those)? ?(pills|tablets|medicine|meds)\b",
    # Risk to the person being cared for
    r"\b(scared|afraid|worried) (that )?i('ll| will| might| could)? (hurt|hit|harm|shake|kill)\b",
    r"\bi (hit|slapped|pushed|shook|hurt) (him|her|them|my (mum|mom|mother|dad|father|husband|wife|grandma|grandpa))\b",
    # Cantonese / Traditional Chinese
    r"自殺", r"想死", r"唔想(再)?(生存|活)", r"不想(再)?活", r"輕生", r"了結(自己|生命)", r"傷害自己",
]

_CJK = re.compile(r"[一-鿿]")
_NEGATORS = re.compile(r"\b(not|never|no longer|isn't|aren't|wasn't|don't|doesn't|didn't|no)\b[^.!?,;]{0,12}$")
_CRISIS_RES = [re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS]


@dataclass(frozen=True)
class Signals:
    stress: float = 0.0
    crisis: bool = False
    evidence: list[str] = field(default_factory=list)


class Detector(Protocol):
    name: str

    def detect(self, text: str) -> Signals: ...


def crisis_screen(text: str) -> list[str]:
    return [m.group(0) for r in _CRISIS_RES if (m := r.search(text))]


class KeywordDetector:
    """Counts stress terms (one point each), skipping ones that are negated.

    A crisis also counts as severe stress (3 points, like the LLM detector's top level).
    """

    name = "keyword"

    def __init__(self, terms: list[str] | None = None):
        self._terms = [(t, _term_regex(t)) for t in (terms or STRESS_TERMS)]

    def detect(self, text: str) -> Signals:
        lowered = text.lower().replace("’", "'")
        hits = []
        for term, regex in self._terms:
            m = regex.search(lowered)
            if m and not _NEGATORS.search(lowered[: m.start()]):
                hits.append(term)
        crisis = crisis_screen(lowered)
        stress = len(hits) + (3 if crisis else 0)
        return Signals(stress=float(stress), crisis=bool(crisis), evidence=hits + crisis)


def _term_regex(term: str) -> re.Pattern[str]:
    if _CJK.search(term):
        return re.compile(re.escape(term))
    return re.compile(r"\b" + re.escape(term) + r"\b")


LLM_DETECTOR_PROMPT = """\
You screen messages sent to a support chatbot for family caregivers.
Rate the sender's current stress from 0 to 3:
0 = none (small talk, practical questions, says they are fine),
1 = mild (tired, a bit worried),
2 = clear distress (overwhelmed, not sleeping, can't cope),
3 = severe (hopeless, breaking point).
Set crisis to true if the message suggests any risk that the sender may harm
themselves or the person they care for, including indirect or passive wording.
When unsure about crisis, choose true."""


class LLMDetector:
    """Asks a Claude model for a structured stress/crisis rating.

    The keyword crisis screen still runs, so a model miss can't hide a crisis.
    """

    name = "llm"

    def __init__(self, model: str = "claude-opus-5-5", client=None):
        import anthropic
        from pydantic import BaseModel

        class Assessment(BaseModel):
            stress_level: int
            crisis: bool
            reason: str

        self._schema = Assessment
        self._client = client or anthropic.Anthropic()
        self._model = model

    def detect(self, text: str) -> Signals:
        response = self._client.beta.messages.parse(
            model=self._model,
            max_tokens=2000,
            system=LLM_DETECTOR_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_format=self._schema,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        keyword_crisis = crisis_screen(text.lower())
        result = response.parsed_output
        if result is None:  # refused or unparseable: rely on the keyword screen
            return Signals(stress=0.0, crisis=bool(keyword_crisis), evidence=keyword_crisis)
        level = min(max(result.stress_level, 0), 3)
        return Signals(
            stress=float(level),
            crisis=result.crisis or bool(keyword_crisis),
            evidence=[f"llm: {result.reason}", *keyword_crisis],
        )
