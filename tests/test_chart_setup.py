"""Tests for the multi-timeframe chart reader.

These use synthetic price series so the expected structure is known exactly —
no network, no market-dependent assertions.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.chart_setup import (
    MAX_ZONE_ATR,
    Candidate,
    Level,
    Pivot,
    _merge_candidates,
    _pick_targets,
    _rsi,
    _atr,
    build_setup,
    classify_structure,
    cluster_levels,
    find_pivots,
    position_size,
    read_timeframe,
    TF_SPECS,
)
from schemas import Direction, SetupStatus


def _frame(closes, spread=0.5, volume=1_000_000) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes - spread / 4,
            "High": closes + spread,
            "Low": closes - spread,
            "Close": closes,
            "Volume": np.full(len(closes), volume, dtype=float),
        },
        index=idx,
    )


def _zigzag(legs, points_per_leg=8) -> np.ndarray:
    """Build a series that walks linearly between the given turning points."""
    out = []
    for a, b in zip(legs, legs[1:]):
        out.extend(np.linspace(a, b, points_per_leg, endpoint=False))
    out.append(legs[-1])
    return np.array(out)


# ── Indicators ────────────────────────────────────────────────────────────────


def test_rsi_saturates_on_a_one_way_market():
    rising = _frame(np.arange(100, 160, dtype=float))["Close"]
    assert _rsi(rising).iloc[-1] > 99

    falling = _frame(np.arange(160, 100, -1, dtype=float))["Close"]
    assert _rsi(falling).iloc[-1] < 1


def test_atr_tracks_the_true_range_of_a_flat_series():
    df = _frame(np.full(60, 100.0), spread=1.0)
    # High-Low is a constant 2.0 and there are no gaps, so ATR converges to it.
    assert _atr(df).iloc[-1] == pytest.approx(2.0, abs=0.05)


# ── Structure ─────────────────────────────────────────────────────────────────


def test_find_pivots_locates_the_turning_points():
    df = _frame(_zigzag([100, 120, 110, 130, 118]))
    pivots = find_pivots(df, strength=3)

    highs = [p for p in pivots if p.kind == "H"]
    lows = [p for p in pivots if p.kind == "L"]
    assert highs and lows
    assert highs[0].price == pytest.approx(120, abs=2)
    assert lows[0].price == pytest.approx(110, abs=2)


def test_structure_reads_higher_highs_and_higher_lows_as_an_uptrend():
    up = find_pivots(_frame(_zigzag([100, 120, 110, 135, 125, 150])), strength=3)
    label, score = classify_structure(up)
    assert score == 100.0
    assert label == "HH / HL"

    down = find_pivots(_frame(_zigzag([150, 125, 135, 110, 120, 100])), strength=3)
    label, score = classify_structure(down)
    assert score == -100.0
    assert label == "LH / LL"


def test_structure_is_unknown_without_enough_pivots():
    label, score = classify_structure([Pivot(0, 100.0, "H")])
    assert label == "insufficient"
    assert score == 0.0


# ── Level clustering ──────────────────────────────────────────────────────────


def test_clustering_does_not_chain_distinct_levels_into_one():
    """Pivots spaced just inside the tolerance must not snowball into one level.

    This is the failure mode that produced a single "107-touch" level spanning
    the whole range: bucketing compared each pivot to the previous one, so a
    dense run drifted indefinitely.
    """
    prices = [100 + i * 0.6 for i in range(40)]  # spacing = 0.6 * tolerance
    df = _frame(np.array(prices))
    pivots = [Pivot(i, p, "L") for i, p in enumerate(prices)]

    levels = cluster_levels([("1d", df, pivots)], price=150.0, tolerance=1.0)

    assert len(levels) > 5, "dense pivots collapsed into too few levels"
    assert max(lv.touches for lv in levels) <= 4


def test_clustering_ignores_pivots_older_than_the_lookback():
    df = _frame(np.linspace(100, 200, 400))
    old = Pivot(10, 101.0, "L")
    recent = Pivot(390, 199.0, "H")

    levels = cluster_levels([("1d", df, [old, recent])], price=200.0, tolerance=1.0,
                            lookback_bars=100)
    assert [round(lv.price) for lv in levels] == [199]


# ── Candidate zones ───────────────────────────────────────────────────────────


def test_merging_never_produces_a_zone_wider_than_the_cap():
    atr = 1.0
    chain = [Candidate(100 + i * 0.4, 100 + i * 0.4 + 0.5, ["order block"]) for i in range(20)]
    merged = _merge_candidates(chain, long=True, atr=atr)
    assert merged
    assert max(c.width for c in merged) <= MAX_ZONE_ATR * atr + 1e-9


def test_confluence_raises_a_candidates_score():
    lone = Candidate(99, 100, ["ema band"])
    stacked = Candidate(99, 100, ["ema band", "order block", "support cluster"])
    assert stacked.score > lone.score


# ── Targets ───────────────────────────────────────────────────────────────────


def test_targets_clear_their_r_multiples_and_stay_ordered():
    levels = [Level(price=p, kind="resistance", touches=3) for p in (101.0, 118.0, 140.0)]
    targets, obstacle = _pick_targets(long=True, entry=100.0, atr=2.0, risk=10.0, levels=levels)

    assert targets == sorted(targets)
    assert targets[0] >= 110.0, "target 1 must be at least 1R away"
    assert targets[1] >= 120.0
    # 101 is friction on the way, not a target worth naming.
    assert obstacle == 101.0


def test_targets_fall_back_to_r_multiples_when_no_structure_is_ahead():
    targets, obstacle = _pick_targets(long=True, entry=100.0, atr=2.0, risk=5.0, levels=[])
    assert targets == [105.0, 110.0, 116.0]
    assert obstacle is None


def test_short_targets_descend():
    levels = [Level(price=p, kind="support", touches=3) for p in (88.0, 78.0)]
    targets, _ = _pick_targets(long=False, entry=100.0, atr=2.0, risk=10.0, levels=levels)
    assert targets == sorted(targets, reverse=True)
    assert all(t < 100.0 for t in targets)


# ── End to end ────────────────────────────────────────────────────────────────


def _uptrend_context():
    closes = _zigzag([80, 100, 92, 118, 108, 140, 130, 165], points_per_leg=22)
    return read_timeframe(TF_SPECS["1d"], _frame(closes, spread=2.0))


def test_a_long_setup_is_ordered_stop_below_entry_below_targets():
    ctx = _uptrend_context()
    price = ctx.read.price
    levels = cluster_levels([("1d", ctx.df, ctx.pivots)], price, 0.35 * ctx.read.atr)

    setup = build_setup(
        Direction.LONG, bias_score=60.0, alignment_pct=100.0, price=price,
        entry_ctx=ctx, htf_ctx=None, ltf_ctx=None, levels=levels,
    )

    assert setup.direction is Direction.LONG
    assert setup.status is not SetupStatus.NO_SETUP
    assert setup.stop < setup.entry_low <= setup.entry_high
    assert setup.entry_ref < setup.target_1 < setup.target_2 < setup.target_3
    assert setup.rr_target_1 >= 1.0
    assert setup.risk_per_share > 0
    assert setup.confluences and setup.triggers and setup.invalidation


def test_no_directional_bias_yields_no_setup():
    ctx = _uptrend_context()
    setup = build_setup(
        Direction.NONE, 0.0, 0.0, ctx.read.price, ctx, None, None, []
    )
    assert setup.status is SetupStatus.NO_SETUP
    assert setup.entry_ref is None


def test_read_timeframe_calls_a_clean_uptrend_an_uptrend():
    read = _uptrend_context().read
    assert read.trend == "uptrend"
    assert read.trend_score > 25
    assert read.structure == "HH / HL"


# ── Sizing ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "risk_per_share, account, risk_pct, expected",
    [
        (2.0, 50_000, 1.0, 250),
        (11.41, 50_000, 1.0, 43),
        (0.0, 50_000, 1.0, 0),      # no risk defined -> no size
        (2.0, 0, 1.0, 0),           # no account -> no size
    ],
)
def test_position_size(risk_per_share, account, risk_pct, expected):
    assert position_size(risk_per_share, account, risk_pct) == expected
