"""Multi-timeframe chart analysis — reads price structure the way you'd read a
TradingView chart, then produces one concrete entry setup.

The pipeline per timeframe is:
    swing pivots -> market structure (HH/HL, BOS/CHoCH) -> order blocks & FVGs
    -> trend score

Those reads are combined into a timeframe-weighted bias, and the entry zone is
picked from the highest-confluence unmitigated demand (or supply) area sitting
between price and the protective swing.

Nothing here places orders or sizes positions for you — it describes the chart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from schemas import (
    ChartAnalysis,
    Direction,
    EntrySetup,
    Level,
    SetupStatus,
    TimeframeRead,
    Zone,
)


# ── Timeframe configuration ───────────────────────────────────────────────────


@dataclass(frozen=True)
class TFSpec:
    key: str
    label: str
    interval: str            # yfinance interval used to fetch
    period: str              # yfinance lookback
    weight: float            # influence on the composite bias
    swing: int               # pivot strength (bars each side)
    resample: str | None = None   # pandas rule if we build this TF from a smaller one


TF_SPECS: dict[str, TFSpec] = {
    "1wk": TFSpec("1wk", "Weekly", "1wk", "5y", 3.0, 4),
    "1d": TFSpec("1d", "Daily", "1d", "2y", 3.0, 5),
    "4h": TFSpec("4h", "4-Hour", "1h", "720d", 2.0, 5, resample="4h"),
    "1h": TFSpec("1h", "1-Hour", "1h", "180d", 1.0, 5),
    "15m": TFSpec("15m", "15-Min", "15m", "59d", 0.5, 6),
}

# Swing trading (1 week – 1 month holds): the daily is the decision chart,
# weekly sets the regime, intraday only refines the trigger.
DEFAULT_TIMEFRAMES = ["1wk", "1d", "4h", "1h"]
DEFAULT_ENTRY_TF = "1d"

MIN_BARS = 60


# ── Indicators (pandas-native, Wilder smoothing where it matters) ─────────────


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff().fillna(0.0)
    gain = _rma(delta.clip(lower=0), n)
    loss = _rma((-delta).clip(lower=0), n)
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # With no losses in the window RSI is 100, not undefined; with no movement
    # at all it is neutral. Filling both with 50 would hide one-way markets.
    no_loss = loss <= 0
    return rsi.where(~no_loss, np.where(gain > 0, 100.0, 50.0)).fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _rma(_true_range(df), n)


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _rma(_true_range(df), n)
    plus_di = 100 * _rma(pd.Series(plus_dm, index=df.index), n) / atr.replace(0, np.nan)
    minus_di = 100 * _rma(pd.Series(minus_dm, index=df.index), n) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _rma(dx.fillna(0), n)


def _macd_state(close: pd.Series) -> str:
    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    hist = macd - signal
    if len(hist) < 2:
        return "neutral"
    now, prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    if now > 0:
        return "bullish" if now >= prev else "bullish fading"
    if now < 0:
        return "bearish" if now <= prev else "bearish fading"
    return "neutral"


# ── Structure primitives ──────────────────────────────────────────────────────


@dataclass
class Pivot:
    idx: int
    price: float
    kind: str  # "H" or "L"


def find_pivots(df: pd.DataFrame, strength: int) -> list[Pivot]:
    """Fractal pivots: a high with `strength` lower highs on both sides (and vice versa).

    The last `strength` bars can never be confirmed — that lag is intentional and
    is what keeps the structure read from repainting.
    """
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    pivots: list[Pivot] = []
    for i in range(strength, len(df) - strength):
        window = slice(i - strength, i + strength + 1)
        if highs[i] == highs[window].max() and (highs[window] < highs[i]).sum() >= strength:
            pivots.append(Pivot(i, float(highs[i]), "H"))
        elif lows[i] == lows[window].min() and (lows[window] > lows[i]).sum() >= strength:
            pivots.append(Pivot(i, float(lows[i]), "L"))
    return pivots


def _alternating(pivots: list[Pivot]) -> list[Pivot]:
    """Collapse consecutive same-kind pivots, keeping the most extreme one."""
    out: list[Pivot] = []
    for p in pivots:
        if out and out[-1].kind == p.kind:
            better = p.price > out[-1].price if p.kind == "H" else p.price < out[-1].price
            if better:
                out[-1] = p
        else:
            out.append(p)
    return out


def classify_structure(pivots: list[Pivot]) -> tuple[str, float]:
    """Return (label, score -100..100) from the last two highs and two lows."""
    piv = _alternating(pivots)
    highs = [p for p in piv if p.kind == "H"][-2:]
    lows = [p for p in piv if p.kind == "L"][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return "insufficient", 0.0
    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price > lows[-2].price
    if hh and hl:
        return "HH / HL", 100.0
    if not hh and not hl:
        return "LH / LL", -100.0
    if hh and not hl:
        return "HH / LL (expanding)", 0.0
    return "LH / HL (contracting)", 0.0


def detect_last_event(df: pd.DataFrame, pivots: list[Pivot], strength: int) -> str:
    """Most recent break of structure / change of character, as a label."""
    piv = _alternating(pivots)
    if len(piv) < 2:
        return ""
    closes = df["Close"].to_numpy(dtype=float)
    event = ""
    last_high: Pivot | None = None
    last_low: Pivot | None = None
    trend = 0  # +1 up, -1 down
    for i in range(len(df)):
        # a pivot only becomes usable `strength` bars after it printed
        for p in piv:
            if p.idx + strength == i:
                if p.kind == "H":
                    last_high = p
                else:
                    last_low = p
        if last_high and closes[i] > last_high.price and (i == 0 or closes[i - 1] <= last_high.price):
            event = "CHoCH up" if trend < 0 else "BOS up"
            trend = 1
            last_high = None
        elif last_low and closes[i] < last_low.price and (i == 0 or closes[i - 1] >= last_low.price):
            event = "CHoCH down" if trend > 0 else "BOS down"
            trend = -1
            last_low = None
    return event


# ── Zones: order blocks and fair value gaps ───────────────────────────────────


def find_order_blocks(
    df: pd.DataFrame, tf: str, strength: int, lookback: int = 12, max_zones: int = 6
) -> list[Zone]:
    """Last opposing candle before the impulse that broke structure.

    Zone runs from the candle's wick extreme to its open — the area price left
    behind on the way out, and the area it tends to revisit.
    """
    pivots = _alternating(find_pivots(df, strength))
    if not pivots:
        return []

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    idx = df.index
    atr = _atr(df).to_numpy(dtype=float)

    zones: list[Zone] = []
    last_high: Pivot | None = None
    last_low: Pivot | None = None

    for i in range(len(df)):
        for p in pivots:
            if p.idx + strength == i:
                if p.kind == "H":
                    last_high = p
                else:
                    last_low = p

        bull_break = last_high is not None and c[i] > last_high.price and c[i - 1] <= last_high.price
        bear_break = last_low is not None and c[i] < last_low.price and c[i - 1] >= last_low.price
        if not (bull_break or bear_break) or i == 0:
            continue

        want_down_candle = bull_break
        ob_i = None
        for back in range(1, min(lookback, i) + 1):
            j = i - back
            if want_down_candle and c[j] < o[j]:
                ob_i = j
                break
            if not want_down_candle and c[j] > o[j]:
                ob_i = j
                break
        if ob_i is None:
            continue

        ref_atr = atr[i] if not math.isnan(atr[i]) and atr[i] > 0 else abs(c[i] - c[ob_i]) or 1.0
        impulse = abs(c[i] - c[ob_i]) / ref_atr

        if want_down_candle:
            top, bottom = float(o[ob_i]), float(l[ob_i])
            kind = "bullish_ob"
        else:
            top, bottom = float(h[ob_i]), float(o[ob_i])
            kind = "bearish_ob"
        if top <= bottom:
            continue

        after_low = float(l[ob_i + 1 :].min()) if ob_i + 1 < len(df) else float(l[ob_i])
        after_high = float(h[ob_i + 1 :].max()) if ob_i + 1 < len(df) else float(h[ob_i])
        after_close = c[ob_i + 1 :]
        if kind == "bullish_ob":
            tested = after_low <= top
            broken = bool(len(after_close) and (after_close < bottom).any())
        else:
            tested = after_high >= bottom
            broken = bool(len(after_close) and (after_close > top).any())

        zones.append(
            Zone(
                kind=kind,
                timeframe=tf,
                top=round(top, 4),
                bottom=round(bottom, 4),
                created_at=str(idx[ob_i].date() if hasattr(idx[ob_i], "date") else idx[ob_i]),
                tested=tested,
                broken=broken,
                strength=round(float(impulse), 2),
            )
        )

    # newest first, de-duplicate overlapping zones of the same kind
    zones.reverse()
    kept: list[Zone] = []
    for z in zones:
        if any(k.kind == z.kind and not (z.bottom > k.top or z.top < k.bottom) for k in kept):
            continue
        kept.append(z)
        if len(kept) >= max_zones:
            break
    return kept


def find_fvgs(df: pd.DataFrame, tf: str, max_zones: int = 4, min_atr_frac: float = 0.15) -> list[Zone]:
    """Three-bar imbalances where price moved too fast to trade an area."""
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    idx = df.index
    atr = _atr(df).to_numpy(dtype=float)
    out: list[Zone] = []

    for i in range(len(df) - 1, 1, -1):
        a = atr[i] if not math.isnan(atr[i]) and atr[i] > 0 else 0.0
        if l[i] > h[i - 2] and (a == 0 or (l[i] - h[i - 2]) >= min_atr_frac * a):
            top, bottom, kind = float(l[i]), float(h[i - 2]), "bullish_fvg"
        elif h[i] < l[i - 2] and (a == 0 or (l[i - 2] - h[i]) >= min_atr_frac * a):
            top, bottom, kind = float(l[i - 2]), float(h[i]), "bearish_fvg"
        else:
            continue

        after = slice(i + 1, len(df))
        filled = (
            (float(l[after].min()) <= bottom) if kind == "bullish_fvg" and i + 1 < len(df)
            else (float(h[after].max()) >= top) if i + 1 < len(df)
            else False
        )
        out.append(
            Zone(
                kind=kind,
                timeframe=tf,
                top=round(top, 4),
                bottom=round(bottom, 4),
                created_at=str(idx[i].date() if hasattr(idx[i], "date") else idx[i]),
                tested=bool(filled),
                broken=bool(filled),
                strength=round(float((top - bottom) / a), 2) if a else 0.0,
            )
        )
        if len(out) >= max_zones * 3:
            break
    return out[: max_zones * 2]


def cluster_levels(
    pivot_sets: list[tuple[str, pd.DataFrame, list[Pivot]]],
    price: float,
    tolerance: float,
    lookback_bars: int = 300,
) -> list[Level]:
    """Merge pivots into horizontal levels by proximity.

    Only recent pivots count — a level nobody has traded against in two years is
    not where this move stops.
    """
    raw: list[tuple[float, str, str]] = []
    for tf, df, pivots in pivot_sets:
        idx = df.index
        cutoff = max(0, len(df) - lookback_bars)
        for p in pivots:
            if p.idx < cutoff:
                continue
            stamp = idx[p.idx]
            raw.append((p.price, tf, str(stamp.date() if hasattr(stamp, "date") else stamp)))
    raw.sort(key=lambda r: r[0])

    levels: list[Level] = []
    bucket: list[tuple[float, str, str]] = []

    def flush() -> None:
        if not bucket:
            return
        avg = sum(b[0] for b in bucket) / len(bucket)
        levels.append(
            Level(
                price=round(avg, 2),
                kind="support" if avg < price else "resistance",
                touches=len(bucket),
                timeframes=sorted({b[1] for b in bucket}),
                last_touch=max(b[2] for b in bucket),
            )
        )

    for r in raw:
        # Compare against the bucket's origin, not its last member — otherwise a
        # dense run of pivots chains into one smeared "level" spanning the range.
        if bucket and (r[0] - bucket[0][0] > tolerance or r[0] - bucket[-1][0] > tolerance * 0.6):
            flush()
            bucket = []
        bucket.append(r)
    flush()
    return levels


# ── Per-timeframe read ────────────────────────────────────────────────────────


@dataclass
class TFContext:
    """Everything computed for one timeframe, kept for the setup builder."""
    spec: TFSpec
    df: pd.DataFrame
    read: TimeframeRead
    pivots: list[Pivot] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)


def _ema_stack_score(price: float, e20: float, e50: float, e200: float | None) -> tuple[str, float]:
    ordered_up = price > e20 > e50 and (e200 is None or e50 > e200)
    ordered_dn = price < e20 < e50 and (e200 is None or e50 < e200)
    if ordered_up:
        return "bullish", 100.0
    if ordered_dn:
        return "bearish", -100.0
    score = 0.0
    score += 40 if price > e20 else -40
    score += 30 if price > e50 else -30
    if e200 is not None:
        score += 30 if price > e200 else -30
    return "mixed", max(-100.0, min(100.0, score))


def read_timeframe(spec: TFSpec, df: pd.DataFrame) -> TFContext:
    """Turn one OHLCV frame into a structured read of that timeframe."""
    close = df["Close"]
    price = float(close.iloc[-1])

    e20 = _ema(close, 20)
    e50 = _ema(close, 50)
    e200 = _ema(close, 200) if len(df) >= 200 else None
    ema200_val = float(e200.iloc[-1]) if e200 is not None else None

    rsi = float(_rsi(close).iloc[-1])
    adx = float(_adx(df).iloc[-1])
    atr_series = _atr(df)
    atr = float(atr_series.iloc[-1])
    vol_avg = df["Volume"].rolling(20).mean()
    rel_vol = (
        round(float(df["Volume"].iloc[-1]) / float(vol_avg.iloc[-1]), 2)
        if not pd.isna(vol_avg.iloc[-1]) and float(vol_avg.iloc[-1]) > 0
        else None
    )

    pivots = find_pivots(df, spec.swing)
    structure, struct_score = classify_structure(pivots)
    event = detect_last_event(df, pivots, spec.swing)
    stack_label, stack_score = _ema_stack_score(price, float(e20.iloc[-1]), float(e50.iloc[-1]), ema200_val)
    macd = _macd_state(close)

    momentum = (rsi - 50) * 2
    if macd.startswith("bullish"):
        momentum += 20
    elif macd.startswith("bearish"):
        momentum -= 20
    momentum = max(-100.0, min(100.0, momentum))

    trend_score = 0.40 * struct_score + 0.35 * stack_score + 0.25 * momentum
    # A trendless tape shouldn't vote as loudly as a trending one.
    if adx < 20:
        trend_score *= 0.6
    trend_score = round(max(-100.0, min(100.0, trend_score)), 1)

    if trend_score >= 25:
        trend = "uptrend"
    elif trend_score <= -25:
        trend = "downtrend"
    else:
        trend = "range"

    alt = _alternating(pivots)
    swing_high = next((p.price for p in reversed(alt) if p.kind == "H"), None)
    swing_low = next((p.price for p in reversed(alt) if p.kind == "L"), None)

    notes: list[str] = []
    if adx >= 25:
        notes.append(f"ADX {adx:.0f} — trending")
    elif adx < 18:
        notes.append(f"ADX {adx:.0f} — chop, mean-reversion risk")
    if rsi >= 70:
        notes.append(f"RSI {rsi:.0f} — extended, poor spot to chase")
    elif rsi <= 30:
        notes.append(f"RSI {rsi:.0f} — oversold")
    if rel_vol and rel_vol >= 1.5:
        notes.append(f"Volume {rel_vol}x average")
    if event:
        notes.append(event)

    read = TimeframeRead(
        timeframe=spec.key,
        label=spec.label,
        bars=len(df),
        price=round(price, 2),
        trend=trend,
        trend_score=trend_score,
        structure=structure,
        last_event=event,
        ema_stack=stack_label,
        ema_20=round(float(e20.iloc[-1]), 2),
        ema_50=round(float(e50.iloc[-1]), 2),
        ema_200=round(ema200_val, 2) if ema200_val is not None else None,
        rsi_14=round(rsi, 1),
        macd_signal=macd,
        adx_14=round(adx, 1),
        atr=round(atr, 3),
        atr_pct=round(atr / price * 100, 2) if price else None,
        rel_volume=rel_vol,
        swing_high=round(swing_high, 2) if swing_high else None,
        swing_low=round(swing_low, 2) if swing_low else None,
        notes=notes,
    )

    zones = find_order_blocks(df, spec.key, spec.swing) + find_fvgs(df, spec.key)
    return TFContext(spec=spec, df=df, read=read, pivots=pivots, zones=zones)


# ── Entry setup construction ──────────────────────────────────────────────────

SOURCE_WEIGHT = {
    "order block": 3.0,
    "htf order block": 4.0,
    "fair value gap": 2.0,
    "ema band": 2.0,
    "support cluster": 2.5,
    "resistance cluster": 2.5,
    "prior swing": 2.0,
}

# An entry zone wider than this is a region, not a level — you cannot risk against it.
MAX_ZONE_ATR = 1.5


@dataclass
class Candidate:
    low: float
    high: float
    kinds: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    strength: float = 0.0

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def score(self) -> float:
        return sum(SOURCE_WEIGHT.get(k, 1.0) for k in set(self.kinds)) + min(self.strength, 3.0)


def _trim(c: Candidate, long: bool, atr: float) -> Candidate:
    """Keep the part of a wide zone that price actually reaches first."""
    limit = MAX_ZONE_ATR * atr
    if c.width <= limit:
        return c
    if long:
        c.low = c.high - limit
    else:
        c.high = c.low + limit
    return c


def _merge_candidates(cands: list[Candidate], long: bool, atr: float) -> list[Candidate]:
    """Overlapping zones are one area of interest — but only while they stay tight."""
    limit = MAX_ZONE_ATR * atr
    cands = sorted(cands, key=lambda c: c.high, reverse=not long)
    merged: list[Candidate] = []
    for c in cands:
        hit = None
        for m in merged:
            if c.low > m.high or c.high < m.low:
                continue
            if max(m.high, c.high) - min(m.low, c.low) > limit:
                continue  # merging would smear two distinct areas into one blob
            hit = m
            break
        if hit:
            hit.low = min(hit.low, c.low)
            hit.high = max(hit.high, c.high)
            hit.kinds.extend(c.kinds)
            hit.labels.extend(c.labels)
            hit.strength = max(hit.strength, c.strength)
        else:
            merged.append(Candidate(c.low, c.high, list(c.kinds), list(c.labels), c.strength))
    return merged


def _collect_candidates(
    long: bool,
    price: float,
    atr: float,
    entry_ctx: TFContext,
    htf_ctx: TFContext | None,
    levels: list[Level],
) -> list[Candidate]:
    cands: list[Candidate] = []
    want_ob = "bullish_ob" if long else "bearish_ob"
    want_fvg = "bullish_fvg" if long else "bearish_fvg"

    pairs = [(entry_ctx, False)]
    if htf_ctx is not None:
        pairs.append((htf_ctx, True))
    for ctx, is_htf in pairs:
        for z in ctx.zones:
            if z.broken:
                continue
            if z.kind == want_ob:
                kind = "htf order block" if is_htf else "order block"
                label = f"{ctx.read.label} order block"
            elif z.kind == want_fvg:
                kind = "fair value gap"
                label = f"{ctx.read.label} fair value gap"
            else:
                continue
            cands.append(_trim(Candidate(z.bottom, z.top, [kind], [label], z.strength), long, atr))

    read = entry_ctx.read
    e20, e50 = read.ema_20, read.ema_50
    if e20 and e50:
        band_low, band_high = min(e20, e50), max(e20, e50)
        if band_high - band_low < 0.3 * atr:
            mid = (band_low + band_high) / 2
            band_low, band_high = mid - 0.15 * atr, mid + 0.15 * atr
        cands.append(
            _trim(Candidate(band_low, band_high, ["ema band"], [f"{read.label} EMA20/50 band"]), long, atr)
        )

    for lv in levels:
        want = "support" if long else "resistance"
        if lv.kind != want or lv.touches < 2:
            continue
        cands.append(
            Candidate(
                lv.price - 0.3 * atr,
                lv.price + 0.3 * atr,
                [f"{want} cluster"],
                [f"{want.capitalize()} {lv.price:.2f} ({lv.touches} touches, {'/'.join(lv.timeframes)})"],
                min(lv.touches * 0.3, 2.0),
            )
        )

    swing = read.swing_low if long else read.swing_high
    if swing:
        cands.append(
            Candidate(
                swing - 0.25 * atr,
                swing + 0.25 * atr,
                ["prior swing"],
                [f"{read.label} swing {'low' if long else 'high'} {swing:.2f}"],
            )
        )

    # Keep only zones price can realistically pull back into, within 4 ATR.
    out = []
    for c in _merge_candidates(cands, long, atr):
        near_edge = c.high if long else c.low
        if long and price - 4 * atr <= near_edge <= price * 1.015:
            out.append(c)
        elif not long and price * 0.985 <= near_edge <= price + 4 * atr:
            out.append(c)
    return out


def _pick_targets(
    long: bool, entry: float, atr: float, risk: float, levels: list[Level]
) -> tuple[list[float], float | None]:
    """Targets are the next places the chart says price has to fight — but only
    the ones far enough away to be worth the risk.

    A level 0.4R above entry is friction, not a target. Each target must clear
    1R / 2R / 3R, snapping to real structure when a level sits near that distance.
    Returns (targets, first_obstacle).
    """
    sign = 1 if long else -1
    ahead = sorted(
        {lv.price for lv in levels if sign * (lv.price - entry) > 0.3 * atr and lv.touches >= 2},
        reverse=not long,
    )
    first_obstacle = ahead[0] if ahead else None

    targets: list[float] = []
    for mult in (1.0, 2.0, 3.2):
        floor = entry + sign * mult * risk
        # nearest structural level at or beyond this R multiple, if one is close by
        snap = next(
            (lv for lv in ahead if sign * (lv - floor) >= 0 and abs(lv - floor) <= 1.5 * atr),
            None,
        )
        target = snap if snap is not None else floor
        if targets and sign * (target - targets[-1]) <= 0:
            target = targets[-1] + sign * max(0.5 * atr, 0.5 * risk)
        targets.append(target)

    if first_obstacle is not None and abs(first_obstacle - targets[0]) < 1e-9:
        first_obstacle = None
    return [round(t, 2) for t in targets], (round(first_obstacle, 2) if first_obstacle else None)


def build_setup(
    bias: Direction,
    bias_score: float,
    alignment_pct: float,
    price: float,
    entry_ctx: TFContext,
    htf_ctx: TFContext | None,
    ltf_ctx: TFContext | None,
    levels: list[Level],
    max_risk_atr: float = 2.5,
) -> EntrySetup:
    if bias is Direction.NONE:
        return EntrySetup(
            status=SetupStatus.NO_SETUP,
            setup_type="No directional edge",
            invalidation="Timeframes disagree — wait for the daily to pick a side.",
        )

    long = bias is Direction.LONG
    read = entry_ctx.read
    atr = read.atr or (price * 0.02)

    candidates = _collect_candidates(long, price, atr, entry_ctx, htf_ctx, levels)
    if not candidates:
        return EntrySetup(
            direction=bias,
            status=SetupStatus.NO_SETUP,
            setup_type="Extended — no defined entry zone within 4 ATR",
            invalidation="Price is far from any structure worth risking against.",
            timeframe_of_entry=read.timeframe,
        )

    # Confluence first, but a great zone 3 ATR away is still a wait, and a wide
    # zone is a worse risk unit than a tight one.
    def rank(c: Candidate) -> float:
        edge = c.high if long else c.low
        distance_atr = min(abs(price - edge) / atr, 4.0)
        return c.score - 1.8 * distance_atr - 0.8 * (c.width / atr)

    best = max(candidates, key=rank)
    entry_low, entry_high = round(best.low, 2), round(best.high, 2)
    # For a long, price falls into the zone — first touch is the top edge.
    first_touch = entry_high if long else entry_low
    inside = entry_low <= price <= entry_high
    entry_ref = round(price if inside else first_touch, 2)

    # Stop sits beyond the zone plus an ATR buffer, and beyond the protective swing.
    buffer = 0.5 * atr
    if long:
        stop = entry_low - buffer
        if read.swing_low and entry_low - read.swing_low <= 1.5 * atr:
            stop = min(stop, read.swing_low - 0.25 * atr)
        stop = max(stop, entry_ref - max_risk_atr * atr)
    else:
        stop = entry_high + buffer
        if read.swing_high and read.swing_high - entry_high <= 1.5 * atr:
            stop = max(stop, read.swing_high + 0.25 * atr)
        stop = min(stop, entry_ref + max_risk_atr * atr)
    stop = round(stop, 2)

    risk = abs(entry_ref - stop)
    if risk <= 0:
        risk = atr
    targets, first_obstacle = _pick_targets(long, entry_ref, atr, risk, levels)
    rr1 = round(abs(targets[0] - entry_ref) / risk, 2)
    rr2 = round(abs(targets[1] - entry_ref) / risk, 2)

    distance_pct = round((entry_ref - price) / price * 100, 2)
    if inside:
        status = SetupStatus.ACTIVE
    elif abs(price - first_touch) <= atr:
        status = SetupStatus.APPROACHING
    else:
        status = SetupStatus.WAIT

    kinds = sorted(set(best.kinds))
    setup_type = _name_setup(long, kinds, read)

    confluences = sorted(set(best.labels))
    confluences.append(f"{read.label} structure {read.structure}")
    if htf_ctx:
        confluences.append(f"{htf_ctx.read.label} {htf_ctx.read.trend} ({htf_ctx.read.trend_score:+.0f})")
    if read.rel_volume and read.rel_volume >= 1.3:
        confluences.append(f"Volume {read.rel_volume}x average")

    notes: list[str] = []
    if first_obstacle is not None:
        notes.append(
            f"First {'overhead supply' if long else 'support'} at {first_obstacle:.2f} — "
            f"expect a pause before target 1; consider trimming there."
        )
    if risk / entry_ref > 0.06:
        notes.append(
            f"Wide stop: {risk / entry_ref * 100:.1f}% of price. Size down or wait for a "
            f"lower-timeframe trigger to tighten it."
        )
    if read.rsi_14 and ((long and read.rsi_14 > 70) or (not long and read.rsi_14 < 30)):
        notes.append(f"{read.label} RSI {read.rsi_14:.0f} — entering against a stretched tape.")

    triggers = _build_triggers(long, entry_low, entry_high, ltf_ctx, read)
    invalidation = (
        f"{read.label} close {'below' if long else 'above'} {stop:.2f} "
        f"({'loses' if long else 'reclaims'} the {entry_low:.2f}–{entry_high:.2f} zone)."
    )

    confidence = _score_confidence(bias_score, alignment_pct, kinds, rr1, read, status)
    setup = EntrySetup(
        direction=bias,
        setup_type=setup_type,
        status=status,
        entry_low=entry_low,
        entry_high=entry_high,
        entry_ref=entry_ref,
        stop=stop,
        target_1=targets[0],
        target_2=targets[1],
        target_3=targets[2],
        rr_target_1=rr1,
        rr_target_2=rr2,
        risk_per_share=round(risk, 2),
        distance_to_entry_pct=distance_pct,
        confidence=confidence,
        grade=_grade(confidence),
        confluences=confluences,
        triggers=triggers,
        invalidation=invalidation,
        timeframe_of_entry=read.timeframe,
        first_obstacle=first_obstacle,
        notes=notes,
    )
    return setup


def _name_setup(long: bool, kinds: list[str], read: TimeframeRead) -> str:
    side = "Long" if long else "Short"
    if any("order block" in k for k in kinds):
        base = "pullback into order block"
    elif "fair value gap" in kinds:
        base = "imbalance fill"
    elif any("cluster" in k for k in kinds):
        base = "retest of level"
    elif "ema band" in kinds:
        base = "moving-average pullback"
    else:
        base = "swing retest"
    if read.trend in ("uptrend", "downtrend"):
        base = f"trend-continuation {base}"
    else:
        base = f"range {base}"
    return f"{side} — {base}"


def _build_triggers(
    long: bool, zone_low: float, zone_high: float, ltf_ctx: TFContext | None, read: TimeframeRead
) -> list[str]:
    ltf = ltf_ctx.read.label if ltf_ctx else "lower timeframe"
    side = "bullish" if long else "bearish"
    return [
        f"Price trades into {zone_low:.2f}–{zone_high:.2f}",
        f"{ltf} prints a {side} reversal candle (engulfing / hammer) inside the zone",
        f"{ltf} makes a {'higher low' if long else 'lower high'} — structure turns before you commit",
        f"Volume expands on the {'bounce' if long else 'rejection'}, not on the approach",
        f"{read.label} RSI holds {'above 40' if long else 'below 60'}",
    ]


def _score_confidence(
    bias_score: float,
    alignment_pct: float,
    kinds: list[str],
    rr1: float,
    read: TimeframeRead,
    status: SetupStatus,
) -> float:
    conf = 0.0
    conf += min(30.0, abs(bias_score) * 0.30)          # multi-timeframe conviction
    conf += alignment_pct * 0.15                        # how unanimous the read is
    conf += min(20.0, 6.0 * len(kinds))                 # confluence at the zone
    conf += min(20.0, max(0.0, (rr1 - 1.0) * 8.0))      # payoff on the first target

    room = 0.0
    if read.adx_14 and read.adx_14 >= 25:
        room += 5
    if read.rsi_14 is not None:
        if 40 <= read.rsi_14 <= 65:
            room += 5
        elif read.rsi_14 > 75 or read.rsi_14 < 25:
            room -= 3
    if read.rel_volume and read.rel_volume >= 1.3:
        room += 2
    conf += room

    if status is SetupStatus.WAIT:
        conf -= 4  # a good plan you can't act on yet
    return round(max(0.0, min(100.0, conf)), 1)


def _grade(conf: float) -> str:
    if conf >= 80:
        return "A+"
    if conf >= 70:
        return "A"
    if conf >= 58:
        return "B"
    if conf >= 45:
        return "C"
    return "D"


# ── Orchestration ─────────────────────────────────────────────────────────────


def fetch_timeframes(ticker: str, timeframes: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Pull every requested timeframe, resampling where yfinance has no native bar."""
    from data.yahoo import get_ohlcv, resample_ohlcv

    frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    for key in timeframes:
        spec = TF_SPECS.get(key)
        if spec is None:
            warnings.append(f"Unknown timeframe '{key}' — skipped")
            continue
        try:
            cache_key = (spec.interval, spec.period)
            if cache_key not in cache:
                cache[cache_key] = get_ohlcv(ticker, spec.interval, spec.period)
            df = cache[cache_key]
            if spec.resample:
                df = resample_ohlcv(df, spec.resample)
            if len(df) < MIN_BARS:
                warnings.append(f"{spec.label}: only {len(df)} bars — skipped (need {MIN_BARS})")
                continue
            frames[key] = df
        except Exception as exc:  # noqa: BLE001 — one bad timeframe shouldn't kill the read
            warnings.append(f"{spec.label}: {exc}")
    return frames, warnings


def analyze_chart(
    ticker: str,
    timeframes: list[str] | None = None,
    entry_tf: str = DEFAULT_ENTRY_TF,
    frames: dict[str, pd.DataFrame] | None = None,
    warnings: list[str] | None = None,
) -> ChartAnalysis:
    """Read `ticker` across timeframes and return one actionable setup.

    This is a description of price structure, not investment advice — the setup
    is what the chart implies, and sizing and the decision to trade are yours.
    """
    ticker = ticker.upper().strip()
    timeframes = timeframes or list(DEFAULT_TIMEFRAMES)
    if frames is None:
        frames, warnings = fetch_timeframes(ticker, timeframes)
    else:
        frames = dict(frames)
        warnings = list(warnings or [])
    if not frames:
        raise ValueError(f"No usable price data for {ticker} on {', '.join(timeframes)}")

    if entry_tf not in frames:
        entry_tf = max(frames, key=lambda k: TF_SPECS[k].weight)
        warnings.append(f"Entry timeframe unavailable — using {TF_SPECS[entry_tf].label}")

    contexts = {k: read_timeframe(TF_SPECS[k], df) for k, df in frames.items()}
    ordered = [contexts[k] for k in timeframes if k in contexts]

    price = float(frames[entry_tf]["Close"].iloc[-1])
    last_stamp = frames[entry_tf].index[-1]

    # Timeframe-weighted bias — the weekly and daily carry the vote for swing holds.
    total_w = sum(c.spec.weight for c in ordered)
    bias_score = round(sum(c.read.trend_score * c.spec.weight for c in ordered) / total_w, 1)
    if bias_score >= 20:
        bias, bias_label = Direction.LONG, "Bullish"
    elif bias_score <= -20:
        bias, bias_label = Direction.SHORT, "Bearish"
    else:
        bias, bias_label = Direction.NONE, "Mixed / no edge"
    if abs(bias_score) >= 60:
        bias_label = f"Strongly {bias_label.lower()}"

    agree = sum(
        c.spec.weight
        for c in ordered
        if (c.read.trend_score > 0) == (bias_score > 0) and abs(c.read.trend_score) >= 10
    )
    alignment = round(agree / total_w * 100, 1)

    entry_ctx = contexts[entry_tf]
    atr = entry_ctx.read.atr or price * 0.02
    # Levels are drawn on the timeframes you'd actually draw them on.
    level_tfs = [c for c in ordered if c.spec.weight >= 3.0] or ordered[:1]
    levels = cluster_levels(
        [(c.spec.key, c.df, c.pivots) for c in level_tfs], price, 0.35 * atr
    )
    levels = [lv for lv in levels if abs(lv.price - price) <= 6 * atr]
    levels.sort(key=lambda lv: (-lv.touches, abs(lv.price - price)))
    levels = levels[:14]

    order = [k for k in TF_SPECS if k in contexts]
    pos = order.index(entry_tf)
    htf_ctx = contexts[order[pos - 1]] if pos > 0 else None
    ltf_ctx = contexts[order[pos + 1]] if pos + 1 < len(order) else None

    setup = build_setup(
        bias, bias_score, alignment, price, entry_ctx, htf_ctx, ltf_ctx, levels
    )

    if entry_ctx.read.atr_pct and entry_ctx.read.atr_pct > 8:
        warnings.append(f"Very volatile: {entry_ctx.read.label} ATR is {entry_ctx.read.atr_pct}% of price")
    conflicting = [c.read.label for c in ordered if abs(c.read.trend_score) >= 30
                   and (c.read.trend_score > 0) != (bias_score > 0)]
    if conflicting:
        warnings.append("Timeframe conflict: " + ", ".join(conflicting) + " disagree with the bias")

    zones = [z for c in ordered if c.spec.weight >= 2.0 for z in c.zones if not z.broken][:12]

    return ChartAnalysis(
        ticker=ticker,
        as_of=str(last_stamp.date() if hasattr(last_stamp, "date") else last_stamp),
        price=round(price, 2),
        bias=bias,
        bias_score=bias_score,
        bias_label=bias_label,
        alignment_pct=alignment,
        timeframes=[c.read for c in ordered],
        setup=setup,
        zones=zones,
        levels=levels,
        warnings=warnings,
    )


def position_size(risk_per_share: float, account: float, risk_pct: float = 1.0) -> int:
    """Shares that keep the loss at `risk_pct` of the account if the stop is hit."""
    if risk_per_share <= 0 or account <= 0:
        return 0
    return int((account * risk_pct / 100) / risk_per_share)
