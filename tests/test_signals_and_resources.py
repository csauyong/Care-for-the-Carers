import pytest

from carebot.resources import Resources
from carebot.signals import KeywordDetector


@pytest.fixture(scope="module")
def detect():
    return KeywordDetector().detect


@pytest.mark.parametrize(
    "text",
    [
        "I'm not stressed at all",
        "The film was intense",
        "I'm uncertain which brand to buy",
        "This heat is killing me",
    ],
)
def test_no_false_stress_or_crisis(detect, text):
    signals = detect(text)
    assert signals.stress == 0 and not signals.crisis


@pytest.mark.parametrize(
    "text",
    [
        "I want to die",
        "Everyone would be better off without me",
        "I've thought about taking all my pills",
        "I'm scared I'll hurt her",
        "我真係好想死",
    ],
)
def test_crisis_detected(detect, text):
    assert detect(text).crisis


def test_curly_apostrophes(detect):
    assert detect("I can’t sleep").stress == 1


def test_unlisted_contacts():
    res = Resources.load()
    assert res.unlisted_contacts("Call 18111 or 182 183, or see https://www.hkada.org.hk/ for groups") == []
    assert res.unlisted_contacts("Try +852 2896 0000") == []
    assert res.unlisted_contacts("Call 2345 6789") == ["2345 6789"]
    assert res.unlisted_contacts("See www.made-up-carers.hk") == ["www.made-up-carers.hk"]
    assert res.unlisted_contacts("It took 3 months, in 2026, at 10:30") == []


def test_blocks_render_every_contact():
    res = Resources.load()
    assert all(c.phone in res.crisis_block() for c in res.crisis_lines)
    assert all((c.phone or c.url) in res.support_block() for c in res.support)
