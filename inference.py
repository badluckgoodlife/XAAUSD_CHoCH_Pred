# ─────────────────────────────────────────────
#  inference.py  –  live prediction on new bars
# ─────────────────────────────────────────────
"""
Usage (after training):
  from inference import predict_latest
  signal = predict_latest()
  # returns dict: { "signal": "LONG" | "SHORT" | "NO_SETUP",
  #                 "probability": 0.73,
  #                 "choch_direction": 1,
  #                 "timestamp": "2024-01-15 08:30:00" }
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import MODEL_PATH, FEATURE_PATH, SYMBOL, INTERVAL, HTF_INTERVAL
from data_fetch import fetch_ohlcv, fetch_htf_bias
from features import build_features
from labeling import get_labeled_dataset


def load_model() -> tuple[XGBClassifier, list]:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    with open(FEATURE_PATH) as f:
        feature_names = [line.strip() for line in f.readlines()]
    return model, feature_names


def predict_latest(threshold: float = 0.55) -> dict:
    """
    Fetches the latest 500 candles, engineers features,
    and returns a signal dict for the most recent CHoCH bar.
    """
    model, feature_names = load_model()

    df_15m = fetch_ohlcv(bars=500)
    df_1h  = fetch_htf_bias(bars=200)
    df     = build_features(df_15m, df_1h)
    df     = df.fillna(0)

    # Get the last row that has a CHoCH
    choch_rows = df[df["choch"] != 0]
    if choch_rows.empty:
        return {"signal": "NO_SETUP", "probability": 0.0,
                "choch_direction": 0, "timestamp": str(df.index[-1])}

    last = choch_rows.iloc[-1]
    X_live = pd.DataFrame([last[feature_names]])
    proba  = model.predict_proba(X_live)[0, 1]

    direction = int(last["choch"])
    if proba >= threshold:
        signal = "LONG" if direction == 1 else "SHORT"
    else:
        signal = "NO_SETUP"

    return {
        "signal":          signal,
        "probability":     round(float(proba), 4),
        "choch_direction": direction,
        "htf_bias":        int(last.get("htf_bias", 0)),
        "session":         _current_session(last),
        "timestamp":       str(last.name),
        "atr":             round(float(last.get("atr", 0)), 4),
    }


def _current_session(row) -> str:
    for s in ["london", "ny", "asia", "overlap"]:
        if row.get(f"session_{s}", 0) == 1:
            return s
    return "off-hours"


# ── Quick batch backtest on historical data ──────────────────────

def backtest_signals(df: pd.DataFrame, model: XGBClassifier,
                     feature_names: list, threshold: float = 0.55) -> pd.DataFrame:
    """
    Runs model predictions over all historical CHoCH events.
    Returns a DataFrame of all predictions vs actual labels
    (if 'label' column is present).
    """
    choch_df = df[df["choch"] != 0].copy().fillna(0)
    if choch_df.empty:
        print("[inference] No CHoCH events found in data.")
        return pd.DataFrame()

    X  = choch_df[feature_names]
    proba = model.predict_proba(X.values)[:, 1]
    pred  = (proba >= threshold).astype(int)

    out = pd.DataFrame({
        "timestamp":  choch_df.index,
        "choch_dir":  choch_df["choch"].values,
        "htf_bias":   choch_df.get("htf_bias", pd.Series(0, index=choch_df.index)).values,
        "session":    choch_df.apply(_current_session, axis=1).values,
        "probability":proba,
        "predicted":  pred,
    })

    if "label" in choch_df.columns:
        out["actual"] = choch_df["label"].values
        out["correct"] = (out["predicted"] == out["actual"]).astype(int)
        print(f"[inference] Backtest accuracy: {out['correct'].mean():.1%}  "
              f"({len(out)} setups)")

    return out
