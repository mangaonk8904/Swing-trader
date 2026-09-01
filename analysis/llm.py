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

ANTHROPIC = "anthropic"
OPENROUTER = "openrouter"
GROQ = "groq"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Sent to OpenRouter for attribution on their dashboards. Harmless elsewhere.
_APP_TITLE = "Swing Trader"
_APP_URL = "https://github.com/mangaonk8904/Swing-trader"

# Ordered best-first per provider. Anything the account cannot see is skipped,
# so a wrong guess here costs nothing — resolution checks the live catalogue.
PREFERRED_MODELS: dict[str, list[str]] = {
    # Anthropic first-party. Opus 5 leads: this reads 13F flow and chart
    # structure, where a fluent-but-wrong paragraph is the costly failure mode.
    ANTHROPIC: [
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
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
    """No provider key is available."""


class NoUsableModel(RuntimeError):
    """The provider exposes no chat model we can use."""


def resolve_provider(
    anthropic_key: str = "",
    openrouter_key: str = "",
    groq_key: str = "",
    preference: str = "",
) -> str:
    """Which provider to use. An explicit preference wins if its key exists.

    Order is quality-first: Anthropic direct, then OpenRouter, then Groq. A
    preference naming a provider with no key falls through rather than failing —
    the panel should still work.
    """
    keys = {ANTHROPIC: anthropic_key, OPENROUTER: openrouter_key, GROQ: groq_key}
    preference = (preference or "").strip().lower()
    if preference in keys and keys[preference]:
        return preference
    for provider in (ANTHROPIC, OPENROUTER, GROQ):
        if keys[provider]:
            return provider
    raise NoProviderConfigured(
        "No LLM key configured — set ANTHROPIC_API_KEY, OPENROUTER_API_KEY or GROQ_API_KEY."
    )


def make_client(provider: str, api_key: str, workspace_id: str = ""):
    """A client for the provider. Imported lazily so one SDK can be absent."""
    if provider == ANTHROPIC:
        from anthropic import Anthropic

        # Identity-linked keys are scoped to a workspace and reject every
        # request without this header.
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        return Anthropic(api_key=api_key, default_headers=headers)
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


class Refused(RuntimeError):
    """The model declined the request on safety grounds."""


# Effort and adaptive thinking exist on current Claude models but 400 on older
# ones; model resolution can land anywhere, so the richer request is attempted
# and quietly retried without the extras.
_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}


def _anthropic_text(response) -> str:
    """Join the text blocks, skipping thinking and tool blocks."""
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


def _chat_anthropic(client, model: str, prompt: str, max_tokens: int, effort: str) -> str:
    """One Messages API call.

    Note there is no `temperature`: sampling parameters were removed on the
    current Claude models and sending one returns a 400.
    """
    base = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    attempts = []
    if effort in _EFFORT_LEVELS:
        attempts.append({**base, "output_config": {"effort": effort}})
    attempts.append(base)

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            response = client.messages.create(**kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_error = exc
            continue
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise Refused(
                f"the model declined this request ({getattr(details, 'category', 'unspecified')})"
            )
        return _anthropic_text(response)
    raise last_error  # type: ignore[misc]


def _chat_openai_compatible(
    client, model: str, prompt: str, temperature: float, max_tokens: int, want_json: bool
) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if want_json:
        try:
            response = client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"}
            )
            return response.choices[0].message.content or ""
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # model does not support JSON mode — fall through to plain text

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def chat(
    prompt: str,
    *,
    anthropic_key: str = "",
    openrouter_key: str = "",
    groq_key: str = "",
    provider_preference: str = "",
    workspace_id: str = "",
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    want_json: bool = False,
    effort: str = "medium",
) -> tuple[str, str]:
    """Send one prompt. Returns (text, "provider/model" actually used).

    JSON mode is requested when the provider supports it but never required —
    callers parse defensively, so a model without it still answers.
    """
    provider = resolve_provider(anthropic_key, openrouter_key, groq_key, provider_preference)
    api_key = {
        ANTHROPIC: anthropic_key, OPENROUTER: openrouter_key, GROQ: groq_key
    }[provider]
    client = make_client(provider, api_key, workspace_id)
    resolved = resolve_model(client, provider, model)
    label = f"{provider}/{resolved}"

    if provider == ANTHROPIC:
        return _chat_anthropic(client, resolved, prompt, max_tokens, effort), label
    return _chat_openai_compatible(
        client, resolved, prompt, temperature, max_tokens, want_json
    ), label


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
