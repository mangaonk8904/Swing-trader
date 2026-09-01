"""Groq access for the dashboard's AI panels.

Groq retires model IDs periodically, and a hardcoded ID fails as a 404 that
looks like a broken feature rather than a stale constant. So the model is
resolved at call time from what the account can actually see: an explicit
GROQ_MODEL wins, otherwise the first entry of PREFERRED_MODELS that is
available, otherwise any usable chat model.
"""

from __future__ import annotations

import json
import re

# Ordered best-first. Anything absent from the account is skipped silently.
PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "deepseek-r1-distill-llama-70b",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

# Speech, safety and embedding models cannot answer a chat prompt.
_NON_CHAT = re.compile(r"(whisper|tts|guard|embed|moderation)", re.I)


class NoUsableModel(RuntimeError):
    """The account exposes no chat model we can use."""


def available_chat_models(client) -> list[str]:
    """Chat-capable model IDs visible to this API key."""
    models = client.models.list()
    return sorted(
        m.id for m in getattr(models, "data", []) if m.id and not _NON_CHAT.search(m.id)
    )


def resolve_model(client, override: str = "") -> str:
    """Pick a model that exists, preferring the best one on offer.

    An explicit override is trusted without a lookup — if it is wrong the API
    error names it, which is the behaviour someone setting it would expect.
    """
    if override:
        return override

    usable = available_chat_models(client)
    if not usable:
        raise NoUsableModel("This Groq account exposes no chat-capable models.")
    for preferred in PREFERRED_MODELS:
        if preferred in usable:
            return preferred
    return usable[0]


def make_client(api_key: str):
    from groq import Groq  # imported lazily so the app runs without groq installed

    return Groq(api_key=api_key)


def chat(
    api_key: str,
    prompt: str,
    *,
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    want_json: bool = False,
) -> tuple[str, str]:
    """Send one prompt. Returns (text, model_used).

    JSON mode is requested when asked for but not required: models that reject
    `response_format` are retried without it, since the caller parses the reply
    defensively anyway.
    """
    client = make_client(api_key)
    resolved = resolve_model(client, model)

    kwargs = {
        "model": resolved,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if want_json:
        try:
            response = client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"}
            )
            return response.choices[0].message.content, resolved
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # model does not support JSON mode — fall through to plain text

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content, resolved


def parse_json_object(text: str) -> dict:
    """Read a JSON object out of a model reply, fenced or prefaced."""
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
