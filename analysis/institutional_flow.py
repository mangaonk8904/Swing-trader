"""Aggregate 13F/NPORT holdings into a read of who is accumulating a stock.

Fintel's ownership endpoint returns the largest holders and how each one moved
last quarter. It carries no classification, so institution type is inferred —
by an LLM when one is configured, by name heuristics otherwise. Every number in
the summary itself is computed, not generated: the model only labels and
narrates, it never supplies a figure.
"""

from __future__ import annotations

import json
import re

from schemas import HolderCategory, HolderMove, InstitutionalFlow

# Keyword fallback used when no LLM is available, and as a sanity net over its
# output. Order matters — the first match wins.
_HEURISTICS: list[tuple[re.Pattern, HolderCategory]] = [
    (re.compile(r"\b(index|etf|s&p 500|spdr|ishares|qqq|total stock market)\b", re.I), HolderCategory.INDEX),
    (re.compile(r"\b(vanguard|blackrock|state street|geode|northern trust|schwab investment)\b", re.I),
     HolderCategory.INDEX),
    (re.compile(r"\b(norges|pension|retirement|sovereign|teachers|calpers)\b", re.I), HolderCategory.PENSION),
    (re.compile(r"\b(jpmorgan|morgan stanley|goldman|bank of america|ubs|citigroup|wells fargo|bnp|barclays|"
                r"deutsche|hsbc|royal bank|td |bank\b)", re.I), HolderCategory.BANK),
    (re.compile(r"\b(fmr|fidelity|t rowe|price t rowe|capital research|invesco|franklin|janus|dodge & cox|"
                r"american funds|pimco)\b", re.I), HolderCategory.MUTUAL_FUND),
    (re.compile(r"\b(capital management|partners|asset management|advisors|lp\b|l\.p\.)", re.I),
     HolderCategory.HEDGE_FUND),
]

# Fund-share-class rows arrive as "VFINX - VANGUARD INDEX FUNDS - ..." and are
# always a pooled vehicle rather than a manager.
_FUND_ROW = re.compile(r"^[A-Z]{4,5}\s*-\s")


def heuristic_category(name: str) -> HolderCategory:
    """Best-effort classification from the institution's name alone."""
    if _FUND_ROW.match(name):
        return HolderCategory.INDEX
    for pattern, category in _HEURISTICS:
        if pattern.search(name):
            return category
    return HolderCategory.OTHER


def _to_int(val) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def _to_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def parse_holders(rows: list[dict]) -> list[HolderMove]:
    """Turn raw Fintel ownership rows into typed moves."""
    moves: list[HolderMove] = []
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        shares = _to_int(r.get("shares"))
        change = _to_int(r.get("sharesChange"))
        moves.append(
            HolderMove(
                name=name,
                category=heuristic_category(name),
                form_type=r.get("formType") or "",
                as_of=r.get("effectiveDate") or "",
                shares=shares,
                shares_change=change,
                shares_pct_change=_to_float(r.get("sharesPercentChange")),
                ownership_pct=_to_float(r.get("ownershipPercent")),
                value=_to_float(r.get("value")),
                value_change=_to_float(r.get("valueChange")),
                # A position whose entire size arrived this quarter is new.
                is_new_position=shares > 0 and shares == change,
                url=r.get("url") or "",
            )
        )
    return moves


def classify_with_llm(names: list[str], groq_key: str, model: str = "llama-3.3-70b-versatile") -> dict[str, str]:
    """Ask the model to label each institution. Returns {name: category}.

    Any name the model omits, mislabels, or returns in an unknown category is
    left to the heuristic by the caller — a bad label must not become a silent
    fact in the summary.
    """
    if not groq_key or not names:
        return {}

    allowed = [c.value for c in HolderCategory]
    prompt = (
        "Classify each investment institution below by type.\n\n"
        f"Allowed categories (use these exact strings): {', '.join(allowed)}\n\n"
        "Guidance:\n"
        "- 'index/passive' covers index funds, ETFs, and the passive arms of Vanguard/BlackRock/"
        "State Street/Geode.\n"
        "- 'hedge fund' means a discretionary or quantitative private fund (Citadel, Millennium, "
        "Point72, Renaissance, Elliott, Tiger). Do NOT label an index provider or a retail asset "
        "manager as a hedge fund.\n"
        "- 'mutual fund' covers long-only public fund managers (Fidelity/FMR, T. Rowe, Capital Research).\n"
        "- 'bank/broker' covers bank and broker-dealer asset management arms.\n"
        "- 'pension/sovereign' covers pension plans and sovereign wealth funds (Norges Bank).\n"
        "- If unsure, use 'other'. Do not guess.\n\n"
        "Return ONLY a JSON object mapping each name exactly as given to its category.\n\n"
        "Names:\n" + "\n".join(f"- {n}" for n in names)
    )

    from groq import Groq  # imported lazily so the app starts without groq installed

    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    raw = json.loads(response.choices[0].message.content)
    valid = {c.value for c in HolderCategory}
    return {k: v for k, v in raw.items() if isinstance(v, str) and v in valid}


def apply_categories(moves: list[HolderMove], labels: dict[str, str]) -> list[HolderMove]:
    """Overlay LLM labels onto moves, keeping the heuristic where none applies."""
    for m in moves:
        label = labels.get(m.name)
        if label:
            try:
                m.category = HolderCategory(label)
            except ValueError:
                pass  # unknown string — heuristic stands
    return moves


def summarize(ticker: str, moves: list[HolderMove], top_n: int = 5) -> InstitutionalFlow:
    """Compute the flow summary. Every figure here is derived, never generated."""
    flow = InstitutionalFlow(ticker=ticker.upper())
    if not moves:
        flow.caveats.append("No 13F/NPORT holdings returned for this ticker.")
        return flow

    flow.holders = len(moves)
    flow.buyers = sum(1 for m in moves if m.shares_change > 0)
    flow.sellers = sum(1 for m in moves if m.shares_change < 0)
    flow.unchanged = flow.holders - flow.buyers - flow.sellers
    flow.new_positions = sum(1 for m in moves if m.is_new_position)
    flow.net_shares = sum(m.shares_change for m in moves)
    flow.net_value = sum(m.value_change or 0.0 for m in moves)

    movers = flow.buyers + flow.sellers
    flow.buy_breadth = round(flow.buyers / movers * 100, 1) if movers else 0.0

    # Sentiment blends how many holders added with how much stock moved, so a
    # single large buyer cannot outvote broad distribution (or the reverse).
    total_shares = sum(m.shares for m in moves)
    intensity = (flow.net_shares / total_shares * 100) if total_shares else 0.0
    breadth_component = (flow.buy_breadth - 50) * 2          # -100..100
    intensity_component = max(-100.0, min(100.0, intensity * 20))
    flow.sentiment_score = round(0.6 * breadth_component + 0.4 * intensity_component, 1)

    score = flow.sentiment_score
    if score >= 40:
        flow.sentiment_label = "Accumulating"
    elif score >= 15:
        flow.sentiment_label = "Mild accumulation"
    elif score > -15:
        flow.sentiment_label = "Mixed / balanced"
    elif score > -40:
        flow.sentiment_label = "Mild distribution"
    else:
        flow.sentiment_label = "Distributing"

    flow.top_buyers = sorted(
        [m for m in moves if m.shares_change > 0], key=lambda m: m.shares_change, reverse=True
    )[:top_n]
    flow.top_sellers = sorted(
        [m for m in moves if m.shares_change < 0], key=lambda m: m.shares_change
    )[:top_n]
    flow.hedge_funds = [m for m in moves if m.category is HolderCategory.HEDGE_FUND]

    by_cat: dict[str, float] = {}
    for m in moves:
        by_cat[m.category.value] = by_cat.get(m.category.value, 0) + m.shares_change
    flow.by_category = by_cat

    periods: dict[str, int] = {}
    for m in moves:
        if m.as_of:
            periods[m.as_of] = periods.get(m.as_of, 0) + 1
    flow.as_of_periods = dict(sorted(periods.items(), reverse=True))

    flow.caveats = _build_caveats(flow, moves)
    return flow


def _build_caveats(flow: InstitutionalFlow, moves: list[HolderMove]) -> list[str]:
    """State plainly what this summary cannot see."""
    caveats = [
        f"Covers the {flow.holders} largest holders Fintel returns, not the full 13F universe — "
        f"funds that exited entirely do not appear at all."
    ]
    if len(flow.as_of_periods) > 1:
        spread = ", ".join(f"{d} ({n})" for d, n in flow.as_of_periods.items())
        caveats.append(
            f"Filings are as of different dates — {spread} — so these position changes are not "
            f"all measured over the same quarter."
        )
    passive = sum(
        m.shares_change for m in moves if m.category in (HolderCategory.INDEX, HolderCategory.PENSION)
    )
    if flow.net_shares and abs(passive) > abs(flow.net_shares) * 0.5:
        caveats.append(
            "Index, ETF and sovereign holders drive most of the net change. Their buying tracks "
            "fund inflows and index weight rather than a view on the stock."
        )
    if not flow.hedge_funds:
        caveats.append(
            "No holder in this set was classified as a hedge fund, so there is no hedge fund "
            "signal to read here."
        )
    return caveats


def build_narrative_prompt(flow: InstitutionalFlow) -> str:
    """Prompt for the written summary. The figures are supplied, not invented."""
    def fmt(moves: list[HolderMove]) -> str:
        return "\n".join(
            f"  - {m.name} ({m.category.value}): {m.shares_change:+,} shares"
            f"{' — NEW position' if m.is_new_position else ''}, as of {m.as_of}"
            for m in moves
        ) or "  (none)"

    return f"""You are summarising institutional 13F/NPORT activity in {flow.ticker} for a swing trader
who holds positions for one week to one month.

These figures are already computed. Use them exactly as given — do not recalculate,
estimate, or introduce any number that is not listed here.

Holders covered: {flow.holders}
Added: {flow.buyers} | Reduced: {flow.sellers} | Unchanged: {flow.unchanged}
New positions opened: {flow.new_positions}
Net share change: {flow.net_shares:+,}
Net value change: ${flow.net_value:+,.0f}
Buy breadth: {flow.buy_breadth}% of movers added
Computed sentiment: {flow.sentiment_label} ({flow.sentiment_score:+.0f} on -100..+100)

Largest adds:
{fmt(flow.top_buyers)}

Largest reductions:
{fmt(flow.top_sellers)}

Hedge funds identified: {', '.join(m.name for m in flow.hedge_funds) or 'none in this set'}

Known limitations of this data:
{chr(10).join('  - ' + c for c in flow.caveats)}

Write 3-4 short paragraphs:
1. What the ownership flow actually shows, naming the notable movers.
2. Whether this is discretionary conviction or mechanical index/passive flow — be explicit,
   since that distinction decides whether the signal means anything.
3. What it implies for a 1-4 week swing position, including how much weight to give it.
   13F data is filed up to 45 days after quarter end, so it is lagging by construction —
   say so if it matters to the read.

Be direct. Do not overstate a weak signal. If the data does not support a conclusion,
say that instead of manufacturing one."""
