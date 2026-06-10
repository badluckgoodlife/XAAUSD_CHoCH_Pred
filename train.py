# ─────────────────────────────────────────────
#  train.py  –  walk-forward + XGBoost
# ─────────────────────────────────────────────
"""
Walk-forward validation:
  ┌──────────────────────────────────────────────────────┐
  │  FOLD 1  │  train  ──────────────  │ test │          │
  │  FOLD 2  │  train  ──────────────────────  │ test │  │
  │  ...                                                  │
  └──────────────────────────────────────────────────────┘

Each fold trains on all data before the test window —
no random shuffling, no future leakage.

Outputs:
  • Per-fold metrics (accuracy, precision, recall, AUC)
  • Feature importance chart saved as PNG
  • Final model trained on 100% of data
  • wf_results.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless — safe for all environments

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from config import (
    N_SPLITS, TEST_SIZE, RANDOM_STATE, EARLY_STOPPING,
    MODEL_PATH, FEATURE_PATH, RESULTS_PATH,
)


# ──────────────────────────────────────────────────────────────────
# WALK-FORWARD SPLITTER
# ──────────────────────────────────────────────────────────────────

def walk_forward_splits(n: int, n_splits: int = N_SPLITS, test_frac: float = TEST_SIZE):
    """
    Generates (train_idx, test_idx) pairs for walk-forward CV.
    Train always expands; test window slides forward.
    """
    test_size  = int(n * test_frac)
    step       = test_size
    splits     = []

    for fold in range(n_splits):
        test_end   = n - fold * step
        test_start = test_end - test_size
        if test_start <= test_size:
            break
        train_idx = np.arange(0, test_start)
        test_idx  = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    # chronological order (earliest test first)
    return list(reversed(splits))


# ──────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────────────────────────

def make_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """
    XGBoost config tuned for tabular financial features:
      - max_depth=5      moderate complexity
      - learning_rate low + many estimators → better generalisation
      - scale_pos_weight handles class imbalance automatically
      - eval_metric=auc  better than logloss for skewed classes
    """
    return XGBClassifier(
        n_estimators       = 600,
        max_depth          = 5,
        learning_rate      = 0.03,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 5,
        gamma              = 0.1,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        scale_pos_weight   = scale_pos_weight,
        eval_metric        = "auc",
        early_stopping_rounds = EARLY_STOPPING,
        random_state       = RANDOM_STATE,
        use_label_encoder  = False,
        verbosity          = 0,
    )


# ──────────────────────────────────────────────────────────────────
# WALK-FORWARD TRAINING
# ──────────────────────────────────────────────────────────────────

def run_walk_forward(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Runs walk-forward CV.  Returns a DataFrame of per-fold metrics.
    """
    X_arr = X.values
    y_arr = y.values

    splits  = walk_forward_splits(len(X_arr))
    results = []

    print(f"\n[train] Walk-forward CV  ({len(splits)} folds)\n{'─'*60}")

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        # handle class imbalance
        pos  = y_tr.sum()
        neg  = len(y_tr) - pos
        spw  = neg / max(pos, 1)

        model = make_model(scale_pos_weight=spw)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False,
        )

        proba  = model.predict_proba(X_te)[:, 1]
        pred   = (proba >= 0.5).astype(int)

        acc   = accuracy_score(y_te, pred)
        prec  = precision_score(y_te, pred, zero_division=0)
        rec   = recall_score(y_te, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, proba)
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(y_te, pred)
        results.append({
            "fold":        fold_i + 1,
            "train_size":  len(train_idx),
            "test_size":   len(test_idx),
            "accuracy":    round(acc,  4),
            "precision":   round(prec, 4),
            "recall":      round(rec,  4),
            "auc":         round(auc,  4),
            "TP": cm[1,1] if cm.shape == (2,2) else 0,
            "FP": cm[0,1] if cm.shape == (2,2) else 0,
            "TN": cm[0,0] if cm.shape == (2,2) else 0,
            "FN": cm[1,0] if cm.shape == (2,2) else 0,
        })
        print(f"  Fold {fold_i+1}  |  "
              f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  auc={auc:.3f}  "
              f"(train={len(train_idx)}, test={len(test_idx)})")

    df_res = pd.DataFrame(results)
    print(f"\n{'─'*60}")
    print(f"  MEAN    |  "
          f"acc={df_res['accuracy'].mean():.3f}  "
          f"prec={df_res['precision'].mean():.3f}  "
          f"rec={df_res['recall'].mean():.3f}  "
          f"auc={df_res['auc'].mean():.3f}")
    print(f"{'─'*60}\n")

    df_res.to_csv(RESULTS_PATH, index=False)
    print(f"[train] Results saved → {RESULTS_PATH}")
    return df_res


# ──────────────────────────────────────────────────────────────────
# FINAL MODEL (trained on all data)
# ──────────────────────────────────────────────────────────────────

def train_final_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """
    Trains on 100% of labeled data.  Use this for live inference.
    """
    print("[train] Training final model on full dataset…")
    pos = y.sum()
    neg = len(y) - pos
    spw = neg / max(pos, 1)

    model = make_model(scale_pos_weight=spw)
    # no eval_set for final model → use full n_estimators
    model.set_params(early_stopping_rounds=None)
    model.fit(X.values, y.values, verbose=False)

    model.save_model(MODEL_PATH)
    print(f"[train] Final model saved → {MODEL_PATH}")

    # save feature names so inference can rebuild X correctly
    with open(FEATURE_PATH, "w") as f:
        f.write("\n".join(X.columns.tolist()))
    print(f"[train] Feature list saved → {FEATURE_PATH}")

    return model


# ──────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ──────────────────────────────────────────────────────────────────

def plot_feature_importance(model: XGBClassifier, feature_names: list, top_n: int = 20):
    """Saves a horizontal bar chart of top-N feature importances."""
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        imp[idx],
        color="#2563EB",
    )
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost gain)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Feature importance plot saved → {path}")
    return path
