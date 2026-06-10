# ─────────────────────────────────────────────
#  data_fetch.py  –  TwelveData loader
# ─────────────────────────────────────────────
"""
Fetches OHLCV data for two timeframes (entry + HTF bias)
from TwelveData.  Returns clean pandas DataFrames.
"""

import time
import requests
import pandas as pd
from config import TWELVEDATA_API_KEY, SYMBOL, INTERVAL, HTF_INTERVAL, BARS


BASE_URL = "https://api.twelvedata.com/time_series"


def _fetch(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    """Raw request wrapper with basic retry logic."""
    params = {
        "symbol":     symbol,
        "interval":   interval,
        "outputsize": min(outputsize, 5000),   # TwelveData hard limit
        "apikey":     TWELVEDATA_API_KEY,
        "format":     "JSON",
        "order":      "ASC",
    }
    for attempt in range(3):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            if "values" not in payload:
                raise ValueError(f"TwelveData error: {payload.get('message', payload)}")
            df = pd.DataFrame(payload["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
        except Exception as e:
            print(f"[fetch] attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to fetch data after 3 attempts")


def fetch_ohlcv(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    bars: int = BARS,
) -> pd.DataFrame:
    print(f"[data] Fetching {bars} × {interval} candles for {symbol}…")
    df = _fetch(symbol, interval, bars)
    print(f"[data] Got {len(df)} rows  |  {df.index[0]} → {df.index[-1]}")
    return df


def fetch_htf_bias(
    symbol: str = SYMBOL,
    interval: str = HTF_INTERVAL,
    bars: int = 1000,
) -> pd.DataFrame:
    """Higher timeframe data used only for trend/bias features."""
    print(f"[data] Fetching HTF ({interval}) bias data…")
    df = _fetch(symbol, interval, bars)
    print(f"[data] HTF: {len(df)} rows  |  {df.index[0]} → {df.index[-1]}")
    return df
