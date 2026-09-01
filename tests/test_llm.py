"""Tests for provider and model resolution.

Providers retire model IDs, and a stale hardcoded ID surfaces as a 404 that
reads like a broken feature. These pin the behaviour that prevents that, plus
the provider selection, using fakes — no network call is made.
"""

import pytest

from analysis.llm import (
    GROQ,
    OPENROUTER,
    NoProviderConfigured,
    NoUsableModel,
    available_chat_models,
    parse_json_object,
    resolve_model,
    resolve_provider,
)


class _Model:
    def __init__(self, mid):
        self.id = mid


class _FakeClient:
    """Stands in for an OpenAI-compatible client, exposing what resolution touches."""

    def __init__(self, model_ids, completions=None):
        self.models = type("M", (), {"list": lambda _self: type(
            "R", (), {"data": [_Model(m) for m in model_ids]})()})()
        self.chat = type("Chat", (), {"completions": completions})()


# ── Provider selection ────────────────────────────────────────────────────────


def test_openrouter_preferred_when_both_keys_present():
    assert resolve_provider(openrouter_key="or", groq_key="gq") == OPENROUTER


def test_falls_back_to_whichever_key_exists():
    assert resolve_provider(openrouter_key="", groq_key="gq") == GROQ
    assert resolve_provider(openrouter_key="or", groq_key="") == OPENROUTER


def test_explicit_preference_wins_when_its_key_is_present():
    assert resolve_provider("or", "gq", preference="groq") == GROQ
    assert resolve_provider("or", "gq", preference="OpenRouter") == OPENROUTER


def test_preference_for_a_provider_without_a_key_is_ignored():
    """Asking for groq with no groq key must not fail — fall back to what works."""
    assert resolve_provider(openrouter_key="or", groq_key="", preference="groq") == OPENROUTER


def test_no_keys_raises_a_named_error():
    with pytest.raises(NoProviderConfigured):
        resolve_provider("", "")


# ── Model filtering ───────────────────────────────────────────────────────────


def test_non_chat_models_are_excluded():
    client = _FakeClient([
        "openai/gpt-4o-mini", "whisper-large-v3", "distil-whisper-large-v3-en",
        "playai-tts", "llama-guard-4-12b", "text-embedding-3", "cohere/rerank-v3",
    ])
    assert available_chat_models(client) == ["openai/gpt-4o-mini"]


# ── Model preference ──────────────────────────────────────────────────────────


def test_best_available_openrouter_preference_wins():
    client = _FakeClient(["openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5", "some/other"])
    assert resolve_model(client, OPENROUTER) == "anthropic/claude-sonnet-4.5"


def test_retired_preferred_model_is_skipped_not_returned():
    """The reported failure: llama-3.3-70b-versatile 404s because the account no
    longer offers it. Resolution must move on rather than insist."""
    client = _FakeClient(["llama-3.1-8b-instant", "gemma2-9b-it"])
    resolved = resolve_model(client, GROQ)
    assert resolved != "llama-3.3-70b-versatile"
    assert resolved == "llama-3.1-8b-instant"


def test_openrouter_auto_is_the_backstop_when_no_named_preference_matches():
    client = _FakeClient(["openrouter/auto", "obscure/model-x"])
    assert resolve_model(client, OPENROUTER) == "openrouter/auto"


def test_unknown_model_still_usable_when_no_preference_matches():
    client = _FakeClient(["some-brand-new-model-2027"])
    assert resolve_model(client, OPENROUTER) == "some-brand-new-model-2027"


def test_explicit_override_is_trusted_without_a_lookup():
    def _boom():
        raise AssertionError("must not list models when an override is given")

    client = _FakeClient([])
    client.models.list = _boom
    assert resolve_model(client, OPENROUTER, "my/pinned-model") == "my/pinned-model"


def test_no_chat_models_raises_a_named_error():
    with pytest.raises(NoUsableModel):
        resolve_model(_FakeClient(["whisper-large-v3"]), GROQ)


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


# ── The chat call path ────────────────────────────────────────────────────────


class _FakeCompletions:
    """Records calls and can simulate a model that rejects JSON mode."""

    def __init__(self, reply="hi", reject_json=False):
        self.reply = reply
        self.reject_json = reject_json
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_json and "response_format" in kwargs:
            raise ValueError("this model does not support response_format")
        message = type("Msg", (), {"content": self.reply})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def _chat_client(model_ids, completions):
    return _FakeClient(model_ids, completions)


def _patched_chat(monkeypatch, client, **kwargs):
    from analysis import llm

    monkeypatch.setattr(llm, "make_client", lambda provider, api_key: client)
    return llm.chat("Say hi", openrouter_key="test-key", **kwargs)


def test_chat_returns_text_and_a_provider_qualified_model_label(monkeypatch):
    completions = _FakeCompletions(reply="the answer")
    client = _chat_client(["anthropic/claude-opus-5"], completions)

    text, label = _patched_chat(monkeypatch, client, max_tokens=42)

    assert text == "the answer"
    assert label == "openrouter/anthropic/claude-opus-5"
    assert completions.calls[0]["max_tokens"] == 42
    assert completions.calls[0]["model"] == "anthropic/claude-opus-5"


def test_json_mode_is_requested_when_asked_for(monkeypatch):
    completions = _FakeCompletions(reply='{"a": 1}')
    client = _chat_client(["anthropic/claude-opus-5"], completions)

    _patched_chat(monkeypatch, client, want_json=True)

    assert completions.calls[0]["response_format"] == {"type": "json_object"}


def test_a_model_rejecting_json_mode_is_retried_as_plain_text(monkeypatch):
    """JSON mode is requested, not required — callers parse defensively, so a
    model without it must still answer rather than surface an error."""
    completions = _FakeCompletions(reply='{"a": 1}', reject_json=True)
    client = _chat_client(["meta-llama/llama-3.3-70b-instruct"], completions)

    text, _ = _patched_chat(monkeypatch, client, want_json=True)

    assert text == '{"a": 1}'
    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]


def test_empty_content_becomes_an_empty_string_not_none(monkeypatch):
    completions = _FakeCompletions(reply=None)
    client = _chat_client(["anthropic/claude-opus-5"], completions)

    text, _ = _patched_chat(monkeypatch, client)

    assert text == ""
