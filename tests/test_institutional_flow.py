"""Tests for the 13F/NPORT institutional flow summary.

The figures in this summary inform position decisions, so they are computed
rather than generated. These tests pin that arithmetic and the fallbacks that
keep it rendering when the LLM is unavailable.
"""

import pytest

from analysis.institutional_flow import (
    apply_categories,
    build_narrative_prompt,
    heuristic_category,
    parse_holders,
    summarize,
)
from schemas import HolderCategory


def _row(name, shares, change, **kw):
    row = {
        "name": name,
        "shares": shares,
        "sharesChange": change,
        "formType": kw.get("form", "13F-HR"),
        "effectiveDate": kw.get("as_of", "2026-06-30"),
        "value": kw.get("value", shares * 10.0),
        "valueChange": kw.get("value_change", change * 10.0),
        "sharesPercentChange": kw.get("pct"),
        "ownershipPercent": kw.get("own"),
        "url": "",
    }
    return row


# ── Classification ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, expected",
    [
        ("VTSMX - VANGUARD INDEX FUNDS - Vanguard Total Stock Market", HolderCategory.INDEX),
        ("Spdr S&p 500 Etf Trust", HolderCategory.INDEX),
        ("BlackRock, Inc.", HolderCategory.INDEX),
        ("State Street Corp", HolderCategory.INDEX),
        ("Norges Bank", HolderCategory.PENSION),
        ("Jpmorgan Chase & Co", HolderCategory.BANK),
        ("Goldman Sachs Group Inc", HolderCategory.BANK),
        ("Fmr Llc", HolderCategory.MUTUAL_FUND),
        ("Price T Rowe Associates Inc /md/", HolderCategory.MUTUAL_FUND),
        ("Sixth Street Partners Management Company, L.P.", HolderCategory.HEDGE_FUND),
    ],
)
def test_heuristic_classification(name, expected):
    assert heuristic_category(name) is expected


def test_index_giants_are_never_called_hedge_funds():
    """The passive arms hold 'capital management' style names — the index rule
    has to win, or every mega cap reports a fake hedge fund signal."""
    for name in ("Vanguard Capital Management Llc", "Geode Capital Management, Llc"):
        assert heuristic_category(name) is HolderCategory.INDEX


def test_llm_labels_override_heuristics_but_junk_does_not():
    moves = parse_holders([_row("Acme Partners", 100, 10)])
    assert moves[0].category is HolderCategory.HEDGE_FUND

    apply_categories(moves, {"Acme Partners": "bank/broker"})
    assert moves[0].category is HolderCategory.BANK

    # An unknown category must leave the existing label alone, not blank it.
    apply_categories(moves, {"Acme Partners": "not-a-real-category"})
    assert moves[0].category is HolderCategory.BANK


# ── Parsing ───────────────────────────────────────────────────────────────────


def test_new_position_detected_when_entire_stake_arrived_this_quarter():
    moves = parse_holders([_row("New Fund", 500, 500), _row("Old Fund", 500, 50)])
    assert moves[0].is_new_position is True
    assert moves[1].is_new_position is False


def test_unparseable_numbers_do_not_raise():
    moves = parse_holders([{"name": "Odd Co", "shares": "n/a", "sharesChange": None}])
    assert moves[0].shares == 0 and moves[0].shares_change == 0


def test_rows_without_a_name_are_dropped():
    assert not parse_holders([{"name": "", "shares": 1}, {"shares": 1}])


# ── Aggregation ───────────────────────────────────────────────────────────────


def test_counts_and_totals():
    flow = summarize("TEST", parse_holders([
        _row("A", 1000, 100), _row("B", 1000, 200),
        _row("C", 1000, -50), _row("D", 1000, 0),
    ]))
    assert (flow.holders, flow.buyers, flow.sellers, flow.unchanged) == (4, 2, 1, 1)
    assert flow.net_shares == 250
    assert flow.net_value == pytest.approx(2500.0)
    assert flow.buy_breadth == pytest.approx(66.7, abs=0.1)


def test_unanimous_accumulation_and_distribution_land_on_opposite_labels():
    buying = summarize("T", parse_holders([_row(f"F{i}", 1000, 300) for i in range(5)]))
    selling = summarize("T", parse_holders([_row(f"F{i}", 1000, -300) for i in range(5)]))

    assert buying.sentiment_score > 40 and buying.sentiment_label == "Accumulating"
    assert selling.sentiment_score < -40 and selling.sentiment_label == "Distributing"


def test_one_large_buyer_cannot_outvote_broad_selling():
    """Sentiment blends breadth with size on purpose: a single whale adding
    while everyone else trims must not read as accumulation."""
    flow = summarize("T", parse_holders(
        [_row("Whale", 10_000, 5_000)] + [_row(f"F{i}", 1_000, -100) for i in range(8)]
    ))
    assert flow.buyers == 1 and flow.sellers == 8
    assert flow.sentiment_score < 0, "breadth should dominate a single large add"


def test_top_movers_are_ranked_by_magnitude():
    flow = summarize("T", parse_holders([
        _row("Small", 1000, 10), _row("Big", 1000, 900), _row("Mid", 1000, 100),
        _row("BigSell", 1000, -800), _row("SmallSell", 1000, -5),
    ]))
    assert [m.name for m in flow.top_buyers] == ["Big", "Mid", "Small"]
    assert [m.name for m in flow.top_sellers] == ["BigSell", "SmallSell"]


def test_empty_input_yields_a_stated_reason_not_a_crash():
    flow = summarize("T", [])
    assert flow.holders == 0
    assert flow.sentiment_label == "No data"
    assert any("No 13F" in c for c in flow.caveats)


# ── Caveats: the summary has to say what it cannot see ────────────────────────


def test_mixed_filing_dates_are_disclosed():
    flow = summarize("T", parse_holders([
        _row("A", 1000, 100, as_of="2026-06-30"),
        _row("B", 1000, 100, as_of="2026-03-31"),
    ]))
    assert any("different dates" in c for c in flow.caveats)
    assert flow.as_of_periods == {"2026-06-30": 1, "2026-03-31": 1}


def test_passive_dominated_flow_is_flagged():
    flow = summarize("T", parse_holders([
        _row("Spdr S&p 500 Etf Trust", 100_000, 90_000),
        _row("Acme Partners", 1_000, 100),
    ]))
    assert any("index weight" in c.lower() or "fund inflows" in c.lower() for c in flow.caveats)


def test_absence_of_hedge_funds_is_stated_explicitly():
    flow = summarize("T", parse_holders([_row("Jpmorgan Chase & Co", 1000, 100)]))
    assert flow.hedge_funds == []
    assert any("no hedge fund signal" in c.lower() for c in flow.caveats)


# ── Narrative prompt ──────────────────────────────────────────────────────────


def test_prompt_carries_the_computed_figures_and_forbids_inventing_more():
    flow = summarize("TEST", parse_holders([_row("Acme Partners", 1000, 250)]))
    prompt = build_narrative_prompt(flow)

    assert "TEST" in prompt
    assert "+250" in prompt
    assert flow.sentiment_label in prompt
    assert "do not recalculate" in prompt.lower()
    # The model must be handed the caveats, not left to discover them.
    for caveat in flow.caveats:
        assert caveat in prompt
