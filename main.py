# ─────────────────────────────────────────────
#  main.py  –  run the full pipeline
# ─────────────────────────────────────────────
"""
Steps:
  1. Fetch 15M + 1H XAUUSD data from TwelveData
  2. Build SMC features
  3. Label CHoCH setups (forward price simulation)
  4. Walk-forward cross-validation
  5. Train final model on all data
  6. Plot feature importances
  7. Print a live signal for the most recent bar

Run:
  python main.py

Optional flags:
  --skip-train   load existing model and just show live signal
  --threshold    prediction confidence threshold (default 0.55)
"""

import argparse
import os
import sys

from data_fetch  import fetch_ohlcv, fetch_htf_bias
from features    import build_features
from labeling    import label_choch_setups, get_labeled_dataset
from train       import run_walk_forward, train_final_model, plot_feature_importance
from inference   import predict_latest, backtest_signals, load_model
from config      import MODEL_PATH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training and load existing model")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Prediction confidence threshold (default: 0.55)")
    p.add_argument("--no-live", action="store_true",
                   help="Skip live signal at the end")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Fetch data ────────────────────────────────────────────
    df_15m = fetch_ohlcv()
    df_1h  = fetch_htf_bias()

    # ── 2. Feature engineering ───────────────────────────────────
    df = build_features(df_15m, df_1h)

    # ── 3. Labeling ──────────────────────────────────────────────
    df = label_choch_setups(df)
    X, y = get_labeled_dataset(df)

    if len(X) < 50:
        print("\n[!] Not enough labeled samples to train (need ≥50). "
              "Try increasing BARS or reducing REWARD_PIPS in config.py")
        sys.exit(1)

    # ── 4 & 5. Train ─────────────────────────────────────────────
    if not args.skip_train:
        wf_results   = run_walk_forward(X, y)
        print("\nWalk-forward summary:")
        print(wf_results[["fold", "accuracy", "precision", "recall", "auc"]].to_string(index=False))

        final_model  = train_final_model(X, y)
        plot_feature_importance(final_model, list(X.columns))
    else:
        if not os.path.exists(MODEL_PATH):
            print(f"[!] No model found at {MODEL_PATH}. Run without --skip-train first.")
            sys.exit(1)
        final_model, feature_names = load_model()
        print(f"[main] Loaded existing model from {MODEL_PATH}")

    # ── 6. Historical backtest ───────────────────────────────────
    print("\n[main] Running backtest on historical CHoCH events…")
    bt = backtest_signals(df, final_model, list(X.columns), threshold=args.threshold)
    if not bt.empty and "actual" in bt.columns:
        print("\nSession breakdown:")
        print(bt.groupby("session")["correct"].agg(["mean", "count"])
                .rename(columns={"mean": "accuracy", "count": "setups"})
                .round(3).to_string())

        print("\nHTF bias breakdown:")
        print(bt.groupby("htf_bias")["correct"].agg(["mean", "count"])
                .rename(columns={"mean": "accuracy", "count": "setups"})
                .round(3).to_string())

    # ── 7. Live signal ───────────────────────────────────────────
    if not args.no_live:
        print("\n[main] Fetching live signal…")
        signal = predict_latest(threshold=args.threshold)
        print("\n" + "═"*40)
        print(f"  SIGNAL     : {signal['signal']}")
        print(f"  Probability: {signal['probability']:.1%}")
        print(f"  CHoCH dir  : {'▲ Bull' if signal['choch_direction']==1 else '▼ Bear'}")
        print(f"  HTF bias   : {signal['htf_bias']}")
        print(f"  Session    : {signal['session']}")
        print(f"  Timestamp  : {signal['timestamp']}")
        print(f"  ATR        : {signal['atr']}")
        print("═"*40 + "\n")


if __name__ == "__main__":
    main()
