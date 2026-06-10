# SMC / ICT Market Pattern Predictor
### XGBoost model trained on ICT-native features — XAUUSD 15M

---

## What this does

Trains a binary classifier to answer:

> *"Given the current market context, will this CHoCH event follow through by at least REWARD_PIPS before hitting RISK_PIPS against?"*

Label = 1 → clean continuation  
Label = 0 → setup failed

---

## File structure

```
smc_ml_pipeline/
├── config.py       ← All tunable settings (API key, R:R, bars, folds)
├── data_fetch.py   ← TwelveData OHLCV loader (15M + 1H)
├── features.py     ← ICT/SMC feature engineering
│     ├── Swing structure (highs/lows, CHoCH, BOS)
│     ├── Liquidity proximity (BSL/SSL distance, ATR-normalised)
│     ├── Order Blocks (bull/bear OB detection + inside-OB flag)
│     ├── Fair Value Gaps (open FVG + distance)
│     ├── Session context (Asia/London/NY killzones, cyclical time)
│     ├── HTF bias (1H structure merged onto 15M, no lookahead)
│     └── Candle anatomy (body ratio, wicks, momentum)
├── labeling.py     ← Forward price simulation → binary labels
├── train.py        ← Walk-forward CV + final XGBoost training
├── inference.py    ← Live prediction + historical backtest
├── main.py         ← Entry point
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.py`:
```python
TWELVEDATA_API_KEY = "your_key_here"
```

---

## Run

**Full pipeline (fetch → train → live signal):**
```bash
python main.py
```

**Load existing model, just show live signal:**
```bash
python main.py --skip-train
```

**Adjust confidence threshold:**
```bash
python main.py --threshold 0.60
```

---

## Key config knobs

| Setting | Default | What it controls |
|---|---|---|
| `REWARD_PIPS` | 15 | Minimum favourable move to label as win |
| `RISK_PIPS` | 8 | Max adverse move before setup is labelled fail |
| `LABEL_LOOKAHEAD` | 10 | How many candles forward to simulate |
| `N_SPLITS` | 5 | Walk-forward folds |
| `BARS` | 5000 | Historical candles to fetch |

Increasing `REWARD_PIPS` → fewer wins (harder label) → lower recall but sharper signals  
Decreasing `RISK_R/PIPS` → more tight stops → more failures in labels

---

## Understanding the outputs

### Walk-forward table
```
fold  accuracy  precision  recall   auc
  1     0.621     0.648     0.571   0.674
  2     0.598     0.601     0.544   0.651
  ...
MEAN    0.610     0.622     0.558   0.661
```
- **AUC > 0.60** = model has real signal. Random = 0.50.
- **Precision** = of signals fired, how many won
- **Recall** = of winning setups, how many were caught

### Live signal output
```
══════════════════════════════════════════
  SIGNAL     : LONG
  Probability: 71.3%
  CHoCH dir  : ▲ Bull
  HTF bias   : 1
  Session    : london
  Timestamp  : 2024-01-15 08:15:00
  ATR        : 2.3400
══════════════════════════════════════════
```
`HTF bias = 1` means 1H structure is bullish — highest quality when aligned.

### Session breakdown (from backtest)
```
          accuracy  setups
session
asia         0.521      47
london       0.671      89   ← best
ny           0.638      76
overlap      0.612      32
```
Filter to London + NY for best results.

---

## Improving the model

1. **Add your own OB/FVG logic** — replace `features.py` functions with your exact SMC rules  
2. **Filter by HTF alignment** — only trade signals where `htf_bias == choch_direction`  
3. **Session filter** — only trade London (07:00–12:00 UTC) open  
4. **Multi-label setup** — expand to 3 classes: win / loss / no-setup  
5. **Add volume** — TwelveData provides tick volume; use it in FVG confirmation  

---

## Notes on financial ML

- Walk-forward CV is mandatory. Random splits will give inflated results.  
- Don't trust AUC > 0.75 without careful overfitting checks.  
- This model predicts *setup quality*, not price. Always respect your own risk rules.  
- Retrain monthly as market conditions shift.
