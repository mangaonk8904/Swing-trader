"""Tests for Groq model resolution.

Groq retires model IDs, and a stale hardcoded ID surfaces as a 404 that reads
like a broken feature. These pin the behaviour that prevents that, using fake
clients — the live API is never called.
"""

import pytest

from analysis.llm import (
    NoUsableModel,
    available_chat_models,
    parse_json_object,
    resolve_model,
)


class _Model:
    def __init__(self, mid):
        self.id = mid


class _FakeClient:
    """Stands in for groq.Groq, exposing only what resolution touches."""

    def __init__(self, model_ids):
        self.models = type("M", (), {"list": lambda _self: type(
            "R", (), {"data": [_Model(m) for m in model_ids]})()})()


# ── Filtering ─────────────────────────────────────────────────────────────────


def test_non_chat_models_are_excluded():
    client = _FakeClient([
        "llama-3.1-8b-instant", "whisper-large-v3", "distil-whisper-large-v3-en",
        "playai-tts", "llama-guard-4-12b", "text-embedding-3",
    ])
    assert available_chat_models(client) == ["llama-3.1-8b-instant"]


# ── Preference order ──────────────────────────────────────────────────────────


def test_best_available_preference_wins():
    client = _FakeClient(["llama-3.1-8b-instant", "openai/gpt-oss-120b", "gemma2-9b-it"])
    # gpt-oss-120b sits above the other two in PREFERRED_MODELS.
    assert resolve_model(client) == "openai/gpt-oss-120b"


def test_retired_preferred_model_is_skipped_not_returned():
    """The exact failure reported: llama-3.3-70b-versatile 404s because the
    account no longer offers it. Resolution must move on rather than insist."""
    client = _FakeClient(["llama-3.1-8b-instant", "gemma2-9b-it"])
    resolved = resolve_model(client)
    assert resolved != "llama-3.3-70b-versatile"
    assert resolved == "llama-3.1-8b-instant"


def test_unknown_model_still_usable_when_no_preference_matches():
    client = _FakeClient(["some-brand-new-model-2027"])
    assert resolve_model(client) == "some-brand-new-model-2027"


def test_explicit_override_is_trusted_without_a_lookup():
    def _boom():
        raise AssertionError("must not list models when an override is given")

    client = _FakeClient([])
    client.models.list = _boom
    assert resolve_model(client, "my-pinned-model") == "my-pinned-model"


def test_no_chat_models_raises_a_named_error():
    with pytest.raises(NoUsableModel):
        resolve_model(_FakeClient(["whisper-large-v3"]))


# ── Reply parsing ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"A": "hedge fund"}', {"A": "hedge fund"}),
        ('```json\n{"A": "index/passive"}\n```', {"A": "index/passive"}),
        ('Sure, here you go:\n{"A": "bank/broker"}\nHope that helps.', {"A": "bank/broker"}),
        ("no json at all", {}),
        ("", {}),
        ("[1, 2, 3]", {}),          # a list is not the object we asked for
        ('{"A": broken', {}),
    ],
)
def test_parse_json_object_survives_chatty_models(text, expected):
    assert parse_json_object(text) == expected
