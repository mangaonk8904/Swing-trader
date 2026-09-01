"""LLM access for the dashboard's AI panels, across providers.

Two things go wrong with a hardcoded `model="..."` string: the provider retires
the ID (a 404 that reads like a broken feature), or you want to move providers
and every call site has to change. So the provider and the model are both
resolved at call time — from whichever API key is configured, and from the
catalogue that key can actually see.

OpenRouter and Groq are both reached through OpenAI-compatible clients, so the
call shape below is identical for either.
"""

from __future__ import annotations

import json
import re

OPENROUTER = "openrouter"
GROQ = "groq"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Sent to OpenRouter for attribution on their dashboards. Harmless elsewhere.
_APP_TITLE = "Swing Trader"
_APP_URL = "https://github.com/mangaonk8904/Swing-trader"

# Ordered best-first per provider. Anything the account cannot see is skipped,
# so a wrong guess here costs nothing — resolution checks the live catalogue.
PREFERRED_MODELS: dict[str, list[str]] = {
    # Verified against OpenRouter's live catalogue. Quality-first: this reads
    # 13F flow and chart structure, where a fluent-but-wrong answer is the
    # failure mode that costs money. Set LLM_MODEL to pin something cheaper.
    OPENROUTER: [
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5",
        "google/gemini-3.5-flash",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat",
        # OpenRouter's own router — a safe backstop when nothing above matches.
        "openrouter/auto",
    ],
    GROQ: [
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
    ],
}

# Speech, safety, embedding and image-output models cannot answer a chat
# prompt; ":batch" variants are the async endpoint, wrong for an interactive UI.
_NON_CHAT = re.compile(r"(whisper|tts|guard|embed|moderation|rerank|-image|:batch)", re.I)


class NoProviderConfigured(RuntimeError):
    """Neither an OpenRouter nor a Groq key is available."""


class NoUsableModel(RuntimeError):
    """The provider exposes no chat model we can use."""


def resolve_provider(openrouter_key: str = "", groq_key: str = "", preference: str = "") -> str:
    """Which provider to use. An explicit preference wins if its key exists."""
    preference = (preference or "").strip().lower()
    if preference == OPENROUTER and openrouter_key:
        return OPENROUTER
    if preference == GROQ and groq_key:
        return GROQ
    if openrouter_key:
        return OPENROUTER
    if groq_key:
        return GROQ
    raise NoProviderConfigured(
        "No LLM key configured — set OPENROUTER_API_KEY or GROQ_API_KEY."
    )


def make_client(provider: str, api_key: str):
    """An OpenAI-compatible client for the provider. Imported lazily."""
    if provider == OPENROUTER:
        from openai import OpenAI

        return OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers={"HTTP-Referer": _APP_URL, "X-Title": _APP_TITLE},
        )
    if provider == GROQ:
        from groq import Groq

        return Groq(api_key=api_key)
    raise ValueError(f"Unknown provider: {provider!r}")


def available_chat_models(client) -> list[str]:
    """Chat-capable model IDs visible to this key."""
    models = client.models.list()
    return sorted(
        m.id for m in getattr(models, "data", []) if m.id and not _NON_CHAT.search(m.id)
    )


def resolve_model(client, provider: str, override: str = "") -> str:
    """Pick a model that exists, preferring the best on offer.

    An explicit override is trusted without a lookup — if it is wrong the API
    error names it, which is what someone setting it would expect to see.
    """
    if override:
        return override

    usable = available_chat_models(client)
    if not usable:
        raise NoUsableModel(f"{provider} exposes no chat-capable models for this key.")
    for preferred in PREFERRED_MODELS.get(provider, []):
        if preferred in usable:
            return preferred
    return usable[0]


def chat(
    prompt: str,
    *,
    openrouter_key: str = "",
    groq_key: str = "",
    provider_preference: str = "",
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    want_json: bool = False,
) -> tuple[str, str]:
    """Send one prompt. Returns (text, "provider/model" actually used).

    JSON mode is requested when asked for but not required: models that reject
    `response_format` are retried without it, since callers parse defensively.
    """
    provider = resolve_provider(openrouter_key, groq_key, provider_preference)
    api_key = openrouter_key if provider == OPENROUTER else groq_key
    client = make_client(provider, api_key)
    resolved = resolve_model(client, provider, model)
    label = f"{provider}/{resolved}"

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
            return (response.choices[0].message.content or ""), label
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # model does not support JSON mode — fall through to plain text

    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or ""), label


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
