# ─────────────────────────────────────────────
#  features.py  –  ICT / SMC feature engineering
# ─────────────────────────────────────────────
"""
All features are computed from raw OHLCV data.
Each function adds columns to the DataFrame in-place.

Features groups:
  A) Price structure  – swing highs/lows, CHoCH, BOS flags
  B) Liquidity zones  – nearest BSL / SSL distance
  C) Order Blocks     – bullish / bearish OB detection
  D) Fair Value Gaps  – FVG presence & fill %
  E) Session context  – killzone flags, time encoding
  F) HTF bias         – 1H trend merged onto 15M
  G) Candle body      – momentum, body/wick ratios
  H) Volatility       – ATR normalisation
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# A)  SWING STRUCTURE
# ──────────────────────────────────────────────────────────────────

def add_swings(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Local pivot highs / lows over a window of ±n bars.
    swing_high = 1 if that candle is a local high
    swing_low  = 1 if that candle is a local low
    """
    highs = df["high"].values
    lows  = df["low"].values
    sh = np.zeros(len(df), dtype=int)
    sl = np.zeros(len(df), dtype=int)
    for i in range(n, len(df) - n):
        if highs[i] == max(highs[i-n:i+n+1]):
            sh[i] = 1
        if lows[i] == min(lows[i-n:i+n+1]):
            sl[i] = 1
    df["swing_high"] = sh
    df["swing_low"]  = sl
    return df


def add_choch_bos(df: pd.DataFrame) -> pd.DataFrame:
    """
    CHoCH (Change of Character): price breaks a previous swing high
    while structure was bearish, or breaks a swing low while bullish.

    BOS (Break of Structure): same direction continuation break.

    Simplified rule used here:
      - Track last confirmed swing high/low
      - If close > last swing high  → potential BOS_bull or CHoCH_bull
        (CHoCH if prior trend was bearish, BOS if bullish)
      - If close < last swing low   → BOS_bear / CHoCH_bear
    """
    closes = df["close"].values
    sh     = df["swing_high"].values
    sl     = df["swing_low"].values

    choch   = np.zeros(len(df), dtype=int)   #  1 = bull CHoCH, -1 = bear CHoCH
    bos     = np.zeros(len(df), dtype=int)   #  1 = bull BOS,   -1 = bear BOS
    trend   = np.zeros(len(df), dtype=int)   #  1 = bullish, -1 = bearish

    last_sh_price = np.nan
    last_sl_price = np.nan
    cur_trend = 0

    for i in range(1, len(df)):
        if sh[i-1]:
            last_sh_price = df["high"].iloc[i-1]
        if sl[i-1]:
            last_sl_price = df["low"].iloc[i-1]

        c = closes[i]
        if not np.isnan(last_sh_price) and c > last_sh_price:
            if cur_trend <= 0:
                choch[i] = 1       # bullish CHoCH
            else:
                bos[i]   = 1       # bullish BOS
            cur_trend = 1

        if not np.isnan(last_sl_price) and c < last_sl_price:
            if cur_trend >= 0:
                choch[i] = -1      # bearish CHoCH
            else:
                bos[i]   = -1      # bearish BOS
            cur_trend = -1

        trend[i] = cur_trend

    df["choch"]         = choch
    df["bos"]           = bos
    df["structure_bias"] = trend   # rolling trend direction
    return df


# ──────────────────────────────────────────────────────────────────
# B)  LIQUIDITY  (BSL / SSL proximity)
# ──────────────────────────────────────────────────────────────────

def add_liquidity(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    BSL = most recent swing high within lookback window  (buyside liq)
    SSL = most recent swing low  within lookback window  (sellside liq)

    Distances are ATR-normalised so they're scale-free.
    """
    highs = df["high"].values
    lows  = df["low"].values
    sh    = df["swing_high"].values
    sl    = df["swing_low"].values
    atr   = df["atr"].values if "atr" in df.columns else np.ones(len(df))

    bsl_dist = np.full(len(df), np.nan)
    ssl_dist = np.full(len(df), np.nan)

    for i in range(lookback, len(df)):
        # last swing high in window
        sh_idx = [j for j in range(i-lookback, i) if sh[j]]
        sl_idx = [j for j in range(i-lookback, i) if sl[j]]
        if sh_idx:
            bsl = highs[sh_idx[-1]]
            bsl_dist[i] = (bsl - df["close"].iloc[i]) / max(atr[i], 1e-9)
        if sl_idx:
            ssl = lows[sl_idx[-1]]
            ssl_dist[i] = (df["close"].iloc[i] - ssl) / max(atr[i], 1e-9)

    df["bsl_dist_atr"] = bsl_dist  # positive = below BSL (not yet swept)
    df["ssl_dist_atr"] = ssl_dist  # positive = above SSL (not yet swept)
    return df


# ──────────────────────────────────────────────────────────────────
# C)  ORDER BLOCKS
# ──────────────────────────────────────────────────────────────────

def add_order_blocks(df: pd.DataFrame, ob_lookback: int = 30) -> pd.DataFrame:
    """
    Bullish OB:  last bearish candle before a strong bullish impulse
                 (impulse = next 3 candles collectively rise > 1.5× ATR)
    Bearish OB:  last bullish candle before a strong bearish impulse

    Features:
      ob_bull_dist_atr  – normalised distance from close to nearest bull OB
      ob_bear_dist_atr  – normalised distance from close to nearest bear OB
      inside_ob_bull    – 1 if current price is inside a bullish OB
      inside_ob_bear    – 1 if current price is inside a bearish OB
    """
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atr    = df["atr"].values if "atr" in df.columns else np.ones(len(df))

    bull_ob_hi = []
    bull_ob_lo = []
    bear_ob_hi = []
    bear_ob_lo = []

    # identify OBs  (look 3 bars ahead for impulse)
    for i in range(1, len(df) - 3):
        impulse_up   = sum(closes[i+k] - opens[i+k] for k in range(1, 4))
        impulse_down = sum(opens[i+k] - closes[i+k] for k in range(1, 4))

        # bullish OB: bearish candle + up impulse
        if closes[i] < opens[i] and impulse_up > 1.5 * atr[i]:
            bull_ob_hi.append((i, highs[i]))
            bull_ob_lo.append((i, lows[i]))

        # bearish OB: bullish candle + down impulse
        if closes[i] > opens[i] and impulse_down > 1.5 * atr[i]:
            bear_ob_hi.append((i, highs[i]))
            bear_ob_lo.append((i, lows[i]))

    ob_bull_dist  = np.full(len(df), np.nan)
    ob_bear_dist  = np.full(len(df), np.nan)
    inside_bull   = np.zeros(len(df), dtype=int)
    inside_bear   = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        c = closes[i]
        # nearest valid bullish OB (price has not traded below it yet)
        valid_bull = [(hi, lo) for (idx, hi), (_, lo) in zip(bull_ob_hi, bull_ob_lo)
                      if idx < i and lo <= c <= hi * 1.002]
        if valid_bull:
            nearest_hi, nearest_lo = valid_bull[-1]
            ob_bull_dist[i] = (c - nearest_lo) / max(atr[i], 1e-9)
            inside_bull[i]  = 1

        valid_bear = [(hi, lo) for (idx, hi), (_, lo) in zip(bear_ob_hi, bear_ob_lo)
                      if idx < i and lo * 0.998 <= c <= hi]
        if valid_bear:
            nearest_hi, nearest_lo = valid_bear[-1]
            ob_bear_dist[i] = (nearest_hi - c) / max(atr[i], 1e-9)
            inside_bear[i]  = 1

    df["ob_bull_dist_atr"] = ob_bull_dist
    df["ob_bear_dist_atr"] = ob_bear_dist
    df["inside_ob_bull"]   = inside_bull
    df["inside_ob_bear"]   = inside_bear
    return df


# ──────────────────────────────────────────────────────────────────
# D)  FAIR VALUE GAPS
# ──────────────────────────────────────────────────────────────────

def add_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bullish FVG: candle[i-1].low  > candle[i+1].high  (gap up)
    Bearish FVG: candle[i-1].high < candle[i+1].low   (gap down)

    fvg_bull_open  – 1 if the most recent bull FVG is still unfilled
    fvg_bear_open  – 1 if the most recent bear FVG is still unfilled
    fvg_bull_dist  – ATR-normalised distance to nearest open bull FVG
    fvg_bear_dist  – ATR-normalised distance to nearest open bear FVG
    """
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atr    = df["atr"].values if "atr" in df.columns else np.ones(len(df))

    bull_fvgs = []   # (top, bottom, index)
    bear_fvgs = []

    for i in range(1, len(df) - 1):
        if lows[i-1] > highs[i+1]:        # gap up
            bull_fvgs.append((lows[i-1], highs[i+1], i))
        if highs[i-1] < lows[i+1]:        # gap down
            bear_fvgs.append((lows[i+1], highs[i-1], i))

    fvg_bull_open  = np.zeros(len(df), dtype=int)
    fvg_bear_open  = np.zeros(len(df), dtype=int)
    fvg_bull_dist  = np.full(len(df), np.nan)
    fvg_bear_dist  = np.full(len(df), np.nan)

    for i in range(2, len(df)):
        c = closes[i]
        # open bull FVGs: price hasn't filled top yet
        open_bull = [(top, bot, idx) for top, bot, idx in bull_fvgs
                     if idx < i - 1 and c > bot]
        if open_bull:
            top, bot, _ = open_bull[-1]
            fvg_bull_open[i] = 1
            fvg_bull_dist[i] = (c - bot) / max(atr[i], 1e-9)

        open_bear = [(top, bot, idx) for top, bot, idx in bear_fvgs
                     if idx < i - 1 and c < top]
        if open_bear:
            top, bot, _ = open_bear[-1]
            fvg_bear_open[i] = 1
            fvg_bear_dist[i] = (top - c) / max(atr[i], 1e-9)

    df["fvg_bull_open"] = fvg_bull_open
    df["fvg_bear_open"] = fvg_bear_open
    df["fvg_bull_dist"] = fvg_bull_dist
    df["fvg_bear_dist"] = fvg_bear_dist
    return df


# ──────────────────────────────────────────────────────────────────
# E)  SESSION / KILLZONE
# ──────────────────────────────────────────────────────────────────

SESSIONS = {
    "asia":     (0,  6),    # 00:00–06:00 UTC
    "london":   (7, 12),    # 07:00–12:00 UTC
    "ny":       (13, 20),   # 13:00–20:00 UTC
    "overlap":  (12, 14),   # London/NY overlap
}

def add_session(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot session flags + cyclical time encoding."""
    hours = df.index.hour
    for name, (start, end) in SESSIONS.items():
        df[f"session_{name}"] = ((hours >= start) & (hours < end)).astype(int)

    # Cyclical encoding so 23:59 and 00:00 are close
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Day of week (avoid Monday gaps)
    dow = df.index.dayofweek   # 0=Mon … 4=Fri
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return df


# ──────────────────────────────────────────────────────────────────
# F)  HTF BIAS  (merge 1H features onto 15M)
# ──────────────────────────────────────────────────────────────────

def add_htf_bias(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Computes structure_bias on 1H data and merges it forward onto 15M
    using merge_asof (no lookahead).
    """
    htf = df_1h.copy()
    htf = add_swings(htf, n=3)
    htf = add_atr(htf)
    htf = add_choch_bos(htf)

    htf_bias = htf[["structure_bias"]].rename(columns={"structure_bias": "htf_bias"})
    # shift by 1 to avoid lookahead from the same candle
    htf_bias = htf_bias.shift(1)

    merged = pd.merge_asof(
        df_15m,
        htf_bias,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    merged["htf_bias"] = merged["htf_bias"].fillna(0).astype(int)
    return merged


# ──────────────────────────────────────────────────────────────────
# G)  CANDLE BODY FEATURES
# ──────────────────────────────────────────────────────────────────

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    body        = df["close"] - df["open"]
    candle_size = df["high"]  - df["low"]

    df["body_ratio"]     = body.abs() / candle_size.replace(0, np.nan)
    df["upper_wick"]     = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_size.replace(0, np.nan)
    df["lower_wick"]     = (df[["open", "close"]].min(axis=1) - df["low"])  / candle_size.replace(0, np.nan)
    df["bull_candle"]    = (body > 0).astype(int)

    # 3-bar momentum
    df["mom_3"]  = df["close"].pct_change(3)
    df["mom_10"] = df["close"].pct_change(10)
    return df


# ──────────────────────────────────────────────────────────────────
# H)  ATR  (used as normaliser everywhere)
# ──────────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    hi, lo, cl = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        hi - lo,
        (hi - cl).abs(),
        (lo - cl).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=period, adjust=False).mean()
    return df


# ──────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ──────────────────────────────────────────────────────────────────

def build_features(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Run all feature steps in the correct dependency order."""
    print("[features] Building feature set…")
    df = df_15m.copy()

    df = add_atr(df)                     # ATR first — used by everything else
    df = add_swings(df)
    df = add_choch_bos(df)
    df = add_liquidity(df)
    df = add_order_blocks(df)
    df = add_fvg(df)
    df = add_session(df)
    df = add_candle_features(df)
    df = add_htf_bias(df, df_1h)

    print(f"[features] Done. Shape: {df.shape}")
    return df


FEATURE_COLS = [
    # Structure
    "swing_high", "swing_low", "choch", "bos", "structure_bias",
    # Liquidity
    "bsl_dist_atr", "ssl_dist_atr",
    # OBs
    "ob_bull_dist_atr", "ob_bear_dist_atr", "inside_ob_bull", "inside_ob_bear",
    # FVGs
    "fvg_bull_open", "fvg_bear_open", "fvg_bull_dist", "fvg_bear_dist",
    # Session
    "session_asia", "session_london", "session_ny", "session_overlap",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # HTF
    "htf_bias",
    # Candle
    "body_ratio", "upper_wick", "lower_wick", "bull_candle", "mom_3", "mom_10",
    # Volatility
    "atr",
]
