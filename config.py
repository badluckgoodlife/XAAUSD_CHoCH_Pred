# ─────────────────────────────────────────────
#  config.py  –  central settings
# ─────────────────────────────────────────────

TWELVEDATA_API_KEY = "YOUR_API_KEY_HERE"   # ← replace this

# Symbol & timeframes
SYMBOL          = "XAU/USD"
INTERVAL        = "15min"          # entry TF
HTF_INTERVAL    = "1h"             # bias TF
BARS            = 5000             # how many candles to fetch

# Labeling
LABEL_LOOKAHEAD = 10               # candles forward to look for the event
REWARD_PIPS     = 15               # minimum favourable move (in price units) to be label=1
RISK_PIPS       = 8                # maximum adverse excursion before invalidation

# Walk-forward
N_SPLITS        = 5                # number of walk-forward folds
TEST_SIZE       = 0.15             # fraction of total data used per test window

# Model
RANDOM_STATE    = 42
EARLY_STOPPING  = 50

# Output paths
MODEL_PATH      = "smc_xgb_model.json"
FEATURE_PATH    = "feature_names.txt"
RESULTS_PATH    = "wf_results.csv"
