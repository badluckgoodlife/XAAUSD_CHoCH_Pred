# ─────────────────────────────────────────────
#  labeling.py  –  SMC setup labeling
# ─────────────────────────────────────────────
"""
Label = 1  when a CHoCH occurs AND price subsequently moves
           REWARD_PIPS in the CHoCH direction before it moves
           RISK_PIPS against it.

Label = 0  when the setup fails (SL hit first or no significant move).

This creates a binary classification problem:
  "Given the current market context, will the next CHoCH lead
   to a clean continuation of at least REWARD_PIPS?"

We only label rows where a CHoCH was detected (choch != 0),
then forward-simulate price to score each one.
"""

import numpy as np
import pandas as pd
from config import LABEL_LOOKAHEAD, REWARD_PIPS, RISK_PIPS


def label_choch_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates over all CHoCH events and assigns binary outcome labels
    using forward price simulation over LABEL_LOOKAHEAD candles.

    Returns the DataFrame with a new 'label' column.
    Rows that are not CHoCH events are set to NaN (filtered out later).
    """
    labels = np.full(len(df), np.nan)

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    choch  = df["choch"].values

    for i in range(len(df) - LABEL_LOOKAHEAD):
        if choch[i] == 0:
            continue   # not a CHoCH candle → skip

        direction = choch[i]   # +1 = bullish CHoCH, -1 = bearish CHoCH
        entry     = closes[i]
        target    = entry + direction * REWARD_PIPS
        stop      = entry - direction * RISK_PIPS

        outcome = 0  # default: failure
        for j in range(i + 1, i + LABEL_LOOKAHEAD + 1):
            hi = highs[j]
            lo = lows[j]

            if direction == 1:   # bullish: want price to hit target
                if hi >= target:
                    outcome = 1
                    break
                if lo <= stop:
                    outcome = 0
                    break
            else:                # bearish: want price to drop to target
                if lo <= target:
                    outcome = 1
                    break
                if hi >= stop:
                    outcome = 0
                    break

        labels[i] = outcome

    df["label"] = labels
    valid = df["label"].notna().sum()
    positive = (df["label"] == 1).sum()
    print(f"[label] Total setups: {valid}  |  Win: {positive}  |  "
          f"Win-rate: {positive/max(valid,1):.1%}")
    return df


def get_labeled_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drops NaN labels and returns (X, y) ready for ML.
    Also fills remaining NaN feature values with 0.
    """
    from features import FEATURE_COLS

    df_valid = df.dropna(subset=["label"]).copy()
    df_valid[FEATURE_COLS] = df_valid[FEATURE_COLS].fillna(0)

    X = df_valid[FEATURE_COLS]
    y = df_valid["label"].astype(int)
    print(f"[label] Dataset: {len(X)} samples  |  "
          f"class balance: {y.mean():.1%} positive")
    return X, y
