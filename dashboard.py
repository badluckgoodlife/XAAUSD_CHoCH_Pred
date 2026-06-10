# ─────────────────────────────────────────────
#  dashboard.py  –  SMC ML Results Dashboard
#  Run with:  streamlit run dashboard.py
# ─────────────────────────────────────────────

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SMC Model Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme tokens ─────────────────────────────────────────────────
NAVY    = "#0A0E1A"
PANEL   = "#111827"
BORDER  = "#1E2D40"
GOLD    = "#F5A623"
GREEN   = "#22C55E"
RED     = "#EF4444"
MUTED   = "#6B7280"
TEXT    = "#E5E7EB"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {{
    background: {NAVY};
    color: {TEXT};
    font-family: 'Inter', sans-serif;
  }}
  [data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
  }}
  .metric-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }}
  .metric-label {{
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 4px;
  }}
  .metric-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: {TEXT};
  }}
  .metric-sub {{
    font-size: 11px;
    color: {MUTED};
    margin-top: 4px;
  }}
  .signal-card {{
    border-radius: 10px;
    padding: 24px 28px;
    text-align: center;
  }}
  .signal-long  {{ background: #052e16; border: 2px solid {GREEN}; }}
  .signal-short {{ background: #2d0a0a; border: 2px solid {RED};   }}
  .signal-none  {{ background: {PANEL};  border: 2px solid {BORDER}; }}
  .signal-label {{
    font-size: 11px; letter-spacing: 0.12em;
    text-transform: uppercase; color: {MUTED};
  }}
  .signal-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 36px; font-weight: 600; margin: 8px 0;
  }}
  .signal-long  .signal-value {{ color: {GREEN}; }}
  .signal-short .signal-value {{ color: {RED};   }}
  .signal-none  .signal-value {{ color: {MUTED}; }}
  .tag {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
  }}
  .tag-bull {{ background: #052e16; color: {GREEN}; border: 1px solid {GREEN}; }}
  .tag-bear {{ background: #2d0a0a; color: {RED};   border: 1px solid {RED};   }}
  .tag-neutral {{ background: {PANEL}; color: {MUTED}; border: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] .stButton button {{
    width: 100%;
  }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter"),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<span style='font-family:IBM Plex Mono;font-size:13px;"
                f"color:{GOLD};font-weight:600;'>◈ SMC MODEL</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    results_path = st.text_input("wf_results.csv path", value="wf_results.csv")
    model_path   = st.text_input("Model path", value="smc_xgb_model.json")
    fi_path      = st.text_input("Feature importance PNG", value="feature_importance.png")

    st.markdown("---")
    threshold = st.slider("Signal threshold", 0.40, 0.80, 0.55, 0.01,
                          help="Minimum model confidence to fire a signal")
    run_live  = st.button("🔄  Refresh live signal", use_container_width=True)
    st.markdown("---")
    st.caption(f"Symbol: **XAU/USD** · 15M")
    st.caption("Walk-forward CV · XGBoost")


# ── Helpers ──────────────────────────────────────────────────────
@st.cache_data
def load_results(path):
    return pd.read_csv(path)

def gauge_color(v, lo=0.5, hi=0.7):
    if v >= hi:   return GREEN
    if v >= lo:   return GOLD
    return RED

def pct(v):
    return f"{v*100:.1f}%"


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(
    f"<h1 style='font-family:IBM Plex Mono;font-size:22px;"
    f"font-weight:600;color:{GOLD};margin-bottom:2px;'>"
    f"SMC Pattern Predictor</h1>"
    f"<p style='color:{MUTED};font-size:13px;margin-top:0;'>"
    f"XAU/USD · 15M · CHoCH Setup Classifier · XGBoost</p>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD RESULTS
# ══════════════════════════════════════════════════════════════════
st.markdown("### Walk-forward Results")

if not os.path.exists(results_path):
    st.warning(f"No results file found at `{results_path}`. Run `python main.py` first.")
    st.stop()

df_res = load_results(results_path)
means  = df_res[["accuracy", "precision", "recall", "auc"]].mean()

# ── Summary metric cards ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, (metric, label, tip) in zip(
    [c1, c2, c3, c4],
    [
        ("auc",       "Mean AUC",       "0.50 = random  ·  0.65+ = useful"),
        ("accuracy",  "Accuracy",       "Overall correct predictions"),
        ("precision", "Precision",      "Of signals fired, % that won"),
        ("recall",    "Recall",         "Of winning setups, % caught"),
    ]
):
    v = means[metric]
    color = gauge_color(v)
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color};">{pct(v)}</div>
          <div class="metric-sub">{tip}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Per-fold chart ───────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    fig = go.Figure()
    metrics_to_plot = [
        ("auc",       GOLD,  "AUC"),
        ("accuracy",  TEXT,  "Accuracy"),
        ("precision", GREEN, "Precision"),
        ("recall",    "#60A5FA", "Recall"),
    ]
    for col, color, name in metrics_to_plot:
        fig.add_trace(go.Scatter(
            x=df_res["fold"], y=df_res[col],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=MUTED,
                  annotation_text="random baseline",
                  annotation_font_color=MUTED)
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Metrics per fold",
                      yaxis=dict(range=[0.3, 1.0], **PLOTLY_LAYOUT["yaxis"]),
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      height=300)
    st.plotly_chart(fig, use_container_width=True)

with right:
    # TP/FP/TN/FN confusion totals across all folds
    if all(c in df_res.columns for c in ["TP","FP","TN","FN"]):
        tp = df_res["TP"].sum()
        fp = df_res["FP"].sum()
        tn = df_res["TN"].sum()
        fn = df_res["FN"].sum()

        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Pred: 0", "Pred: 1"],
            y=["Act: 0",  "Act: 1"],
            text=[[str(tn), str(fp)], [str(fn), str(tp)]],
            texttemplate="%{text}",
            textfont=dict(size=18, family="IBM Plex Mono"),
            colorscale=[[0, PANEL], [1, GOLD]],
            showscale=False,
        ))
        fig_cm.update_layout(**PLOTLY_LAYOUT, title="Confusion (all folds)",
                             height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

# ── Raw fold table ───────────────────────────────────────────────
with st.expander("Raw fold data"):
    display = df_res.copy()
    for c in ["accuracy", "precision", "recall", "auc"]:
        display[c] = display[c].apply(lambda v: f"{v*100:.1f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════
st.markdown("### Feature Importance")

if os.path.exists(fi_path):
    st.image(fi_path, use_column_width=True)
elif os.path.exists(model_path):
    # Re-derive from model if PNG not found
    model = XGBClassifier()
    model.load_model(model_path)
    feature_names_path = model_path.replace(".json", "").replace("smc_xgb_model", "feature_names") + ".txt"
    if os.path.exists("feature_names.txt"):
        with open("feature_names.txt") as f:
            fnames = [l.strip() for l in f]
        imp = model.feature_importances_
        idx = np.argsort(imp)[-20:]
        fig_fi = go.Figure(go.Bar(
            x=imp[idx],
            y=[fnames[i] for i in idx],
            orientation="h",
            marker_color=GOLD,
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, title="Top 20 features", height=500)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Run `python main.py` to generate feature_importance.png")
else:
    st.info("No model or feature importance file found.")


# ══════════════════════════════════════════════════════════════════
# LIVE SIGNAL
# ══════════════════════════════════════════════════════════════════
st.markdown("### Live Signal")

if run_live or "live_signal" not in st.session_state:
    if os.path.exists(model_path):
        with st.spinner("Fetching latest candles…"):
            try:
                import sys, os
                sys.path.insert(0, os.path.dirname(__file__))
                from inference import predict_latest
                st.session_state["live_signal"] = predict_latest(threshold=threshold)
                st.session_state["signal_error"] = None
            except Exception as e:
                st.session_state["live_signal"] = None
                st.session_state["signal_error"] = str(e)
    else:
        st.session_state["live_signal"] = None
        st.session_state["signal_error"] = f"No model at `{model_path}`"

sig = st.session_state.get("live_signal")
err = st.session_state.get("signal_error")

if err:
    st.error(f"Signal error: {err}")
elif sig:
    direction = sig.get("choch_direction", 0)
    signal    = sig.get("signal", "NO_SETUP")
    prob      = sig.get("probability", 0)
    htf       = sig.get("htf_bias", 0)
    session   = sig.get("session", "—")
    ts        = sig.get("timestamp", "—")
    atr_val   = sig.get("atr", 0)

    card_cls  = "signal-long" if signal == "LONG" else ("signal-short" if signal == "SHORT" else "signal-none")
    arrow     = "▲" if signal == "LONG" else ("▼" if signal == "SHORT" else "—")
    htf_tag   = (f'<span class="tag tag-bull">HTF ▲ Bull</span>' if htf == 1
                 else f'<span class="tag tag-bear">HTF ▼ Bear</span>' if htf == -1
                 else f'<span class="tag tag-neutral">HTF Neutral</span>')

    col_sig, col_detail = st.columns([1, 2])

    with col_sig:
        st.markdown(f"""
        <div class="signal-card {card_cls}">
          <div class="signal-label">Current Signal</div>
          <div class="signal-value">{arrow} {signal}</div>
          <div style="font-family:IBM Plex Mono;font-size:22px;color:{MUTED};">{pct(prob)}</div>
          <div style="margin-top:10px;font-size:12px;color:{MUTED};">confidence</div>
        </div>""", unsafe_allow_html=True)

    with col_detail:
        st.markdown(f"""
        <div style="background:{PANEL};border:1px solid {BORDER};border-radius:8px;
                    padding:20px;font-size:13px;line-height:2.2;">
          <div>{htf_tag} &nbsp;
            <span class="tag tag-neutral">{session.upper()}</span>
          </div>
          <div style="margin-top:12px;font-family:IBM Plex Mono;">
            <span style="color:{MUTED};">Timestamp   </span> {ts}<br>
            <span style="color:{MUTED};">ATR         </span> {atr_val:.4f}<br>
            <span style="color:{MUTED};">CHoCH dir   </span>
              {'▲ Bullish' if direction == 1 else '▼ Bearish' if direction == -1 else '—'}<br>
            <span style="color:{MUTED};">Threshold   </span> {threshold:.0%}
          </div>
        </div>""", unsafe_allow_html=True)

    # Probability gauge bar
    bar_color = GREEN if signal == "LONG" else (RED if signal == "SHORT" else MUTED)
    fig_gauge = go.Figure(go.Bar(
        x=[prob], y=[""], orientation="h",
        marker_color=bar_color, width=0.4,
    ))
    fig_gauge.add_vline(x=threshold, line_dash="dash", line_color=GOLD,
                        annotation_text=f"threshold {threshold:.0%}",
                        annotation_font_color=GOLD)
    fig_gauge.update_layout(**PLOTLY_LAYOUT, height=80,
                             xaxis=dict(range=[0, 1], **PLOTLY_LAYOUT["xaxis"]),
                             margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# BACKTEST BREAKDOWN
# ══════════════════════════════════════════════════════════════════
if os.path.exists(model_path) and os.path.exists("feature_names.txt"):
    with st.expander("📊  Session & HTF backtest breakdown (runs inference on historical data)"):
        if st.button("Run backtest breakdown"):
            with st.spinner("Running…"):
                try:
                    from data_fetch import fetch_ohlcv, fetch_htf_bias
                    from features   import build_features
                    from labeling   import label_choch_setups
                    from inference  import backtest_signals, load_model

                    df_15m = fetch_ohlcv(bars=2000)
                    df_1h  = fetch_htf_bias(bars=500)
                    df     = build_features(df_15m, df_1h)
                    df     = label_choch_setups(df)

                    m, fnames = load_model()
                    bt = backtest_signals(df, m, fnames, threshold=threshold)

                    if not bt.empty and "actual" in bt.columns:
                        left2, right2 = st.columns(2)

                        with left2:
                            sess = bt.groupby("session")["correct"].agg(
                                accuracy="mean", setups="count").reset_index()
                            fig_s = px.bar(sess, x="session", y="accuracy",
                                           color="accuracy",
                                           color_continuous_scale=[[0,RED],[0.5,GOLD],[1,GREEN]],
                                           range_color=[0.4, 0.8],
                                           title="Accuracy by session")
                            fig_s.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
                            st.plotly_chart(fig_s, use_container_width=True)

                        with right2:
                            htf_map = {1: "Bull", -1: "Bear", 0: "Neutral"}
                            bt["htf_label"] = bt["htf_bias"].map(htf_map)
                            htf_g = bt.groupby("htf_label")["correct"].agg(
                                accuracy="mean", setups="count").reset_index()
                            fig_h = px.bar(htf_g, x="htf_label", y="accuracy",
                                           color="accuracy",
                                           color_continuous_scale=[[0,RED],[0.5,GOLD],[1,GREEN]],
                                           range_color=[0.4, 0.8],
                                           title="Accuracy by HTF bias")
                            fig_h.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
                            st.plotly_chart(fig_h, use_container_width=True)

                        with st.expander("Raw prediction table"):
                            st.dataframe(bt.head(200), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Backtest error: {e}")
