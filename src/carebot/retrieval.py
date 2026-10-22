"""Retrieves the support services that best fit a conversation.

BM25 over each service's name, description, tags and keywords. English is
split into words; Chinese into character bigrams, which works well for
Cantonese without a word segmenter. Small, dependency-free and deterministic,
so recommendation accuracy can be measured exactly (`carebot eval retrieval`).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .resources import Contact

_WORD = re.compile(r"[a-z0-9']+")
_CJK_RUN = re.compile(r"[一-鿿]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for", "from", "has", "have", "he",
    "her", "him", "his", "how", "i", "i'm", "if", "in", "is", "it", "it's", "just", "me", "my", "of", "on",
    "or", "she", "so", "that", "the", "them", "they", "this", "to", "was", "we", "what", "with", "you", "your",
    "all", "am", "been", "get", "got", "im", "really", "very", "would", "there", "any", "some", "about",
}
# Recommended when nothing in the conversation matches anything specific.
DEFAULT_IDS = ("carer-hotline", "carer-gateway")


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    tokens = [_stem(w) for w in _WORD.findall(text) if w not in _STOPWORDS and len(w) > 1]
    for run in _CJK_RUN.findall(text):
        tokens += [run] if len(run) == 1 else [run[i : i + 2] for i in range(len(run) - 1)]
    return tokens


def _stem(word: str) -> str:
    """Just enough stemming to match plurals and -ing forms."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("sses", "xes", "ches", "shes")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        word = word[:-1]
    if word.endswith("ing") and len(word) - 3 >= 4:
        word = word[:-3]
    return word


class ResourceRetriever:
    def __init__(self, contacts: tuple[Contact, ...] | list[Contact], k1: float = 1.2, b: float = 0.75):
        self.contacts = list(contacts)
        # Tags and keywords are written for retrieval, so they count double.
        self._docs = [
            Counter(tokenize(" ".join([c.name, c.note or ""] + 2 * list(c.tags) + 2 * list(c.keywords))))
            for c in self.contacts
        ]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / max(len(self._docs), 1)
        n = len(self._docs)
        df = Counter(t for d in self._docs for t in d)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[tuple[Contact, float]]:
        terms = tokenize(query)
        results = []
        for contact, doc, length in zip(self.contacts, self._docs, self._lengths):
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg)
                    score += self._idf[t] * tf * (self.k1 + 1) / norm
            results.append((contact, score))
        return sorted(results, key=lambda r: r[1], reverse=True)

    def search(self, query: str, k: int = 3) -> list[Contact]:
        """Top-k matches; falls back to the general carer services if nothing matches."""
        hits = [c for c, score in self.scores(query) if score > 0][:k]
        if not hits:
            index = {c.id: c for c in self.contacts}
            hits = [index[i] for i in DEFAULT_IDS if i in index][:k]
        return hits


LLM_RECOMMENDER_PROMPT = """\
You match family caregivers in Hong Kong to support services. Read what the
caregiver said and pick the services from the list that would help most, best
first. Only use ids from the list. Pick fewer if fewer fit.

Services:
{services}"""


class LLMRecommender:
    """Asks Claude to pick services from the bank: semantic matching, no keywords.

    Same interface as ResourceRetriever, so `carebot eval retrieval --retriever llm`
    benchmarks the two against each other.
    """

    def __init__(self, contacts, model: str = "claude-opus-5-5", client=None):
        import anthropic
        from pydantic import BaseModel

        class Picks(BaseModel):
            service_ids: list[str]

        self.contacts = list(contacts)
        self._index = {c.id: c for c in self.contacts}
        self._schema = Picks
        self._model = model
        self._client = client or anthropic.Anthropic()
        listing = "\n".join(f"- {c.id}: {c.name}. {c.note or ''} Tags: {', '.join(c.tags)}" for c in self.contacts)
        self._system = LLM_RECOMMENDER_PROMPT.format(services=listing)

    def scores(self, query: str) -> list[tuple[Contact, float]]:
        response = self._client.beta.messages.parse(
            model=self._model,
            max_tokens=2000,
            system=self._system,
            messages=[{"role": "user", "content": query}],
            output_format=self._schema,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        picks = response.parsed_output.service_ids if response.parsed_output else []
        ranked = [self._index[i] for i in dict.fromkeys(picks) if i in self._index]
        rest = [c for c in self.contacts if c not in ranked]
        return [(c, float(len(ranked) - i)) for i, c in enumerate(ranked)] + [(c, 0.0) for c in rest]

    def search(self, query: str, k: int = 3) -> list[Contact]:
        hits = [c for c, score in self.scores(query) if score > 0][:k]
        return hits or [self._index[i] for i in DEFAULT_IDS if i in self._index][:k]
