import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime

# ── Page config (unchanged) ────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAANG Predictor",
    page_icon="📈",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLING BLOCK — all CSS lives here, sub-comments label each change
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 900px; }

    .hero {
        background: linear-gradient(135deg, #0f2744 0%, #2563eb 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem 1.8rem;
        margin-bottom: 1.8rem;
        color: white;
    }
    .hero h1 { font-size: 1.85rem; font-weight: 700; margin: 0; color: white !important; }
    .hero p  { font-size: 0.86rem; color: rgba(255,255,255,0.65); margin: 0.35rem 0 0; }

    [data-testid="metric-container"] {
        background: #f7f8fa;
        border: 1px solid #e8eaed;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.15s, box-shadow 0.15s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.09);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem; color: #6b7280;
        font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; color: #111827; }

    div[role="radiogroup"] { gap: 0.45rem; flex-wrap: wrap; }
    div[role="radiogroup"] label {
        background: #f1f5f9 !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 20px !important;
        padding: 0.35rem 1.05rem !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
        cursor: pointer;
        transition: all 0.15s;
    }
    div[role="radiogroup"] label:hover {
        background: #dbeafe !important;
        border-color: #93c5fd !important;
        color: #1d4ed8 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #f8fafc; padding: 0.35rem;
        border-radius: 12px; border-bottom: none !important; gap: 0.4rem;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 0.35rem 1.1rem !important;
        font-weight: 500 !important;
        font-size: 0.87rem !important;
    }

    .info-box {
        background: #f0f7ff;
        border-left: 3px solid #2563eb;
        border-radius: 0 10px 10px 0;
        padding: 0.65rem 1rem;
        font-size: 0.82rem;
        color: #374151;
        margin: 0.4rem 0 1.1rem;
        line-height: 1.55;
    }

    [data-testid="stSlider"] { padding: 0.3rem 0 0.6rem; }

    .footnote { font-size: 0.73rem; color: #9ca3af; margin-top: 0.5rem; line-height: 1.6; }
    hr { border-color: #f0f1f3; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TICKERS = {
    "META":  "Meta (Facebook)",
    "AAPL":  "Apple",
    "AMZN":  "Amazon",
    "NFLX":  "Netflix",
    "GOOG":  "Google",
}

DESCRIPTIONS = {
    "META":  "Parent company of Facebook, Instagram and WhatsApp. One of the world's largest social media and digital advertising platforms.",
    "AAPL":  "Consumer electronics giant behind the iPhone, Mac, iPad and a growing services ecosystem including the App Store and Apple Music.",
    "AMZN":  "Global e-commerce leader and operator of AWS, the world's largest cloud computing platform.",
    "NFLX":  "Streaming pioneer with over 200 million subscribers worldwide, producing original content across film and television.",
    "GOOG":  "Parent company Alphabet operates the world's most-used search engine and the largest digital advertising network.",
}

LOOKBACK   = 100
TRAIN_FRAC = 0.70

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(ticker):
    end   = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)
    df    = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

@st.cache_resource(show_spinner=False)
def load_model_file(ticker):
    return load_model(f"{ticker}_model.keras")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION PIPELINE (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def run_predictions(df, model):
    prices = df[["Adj Close"]].values
    total  = len(prices)
    split  = int(total * TRAIN_FRAC)

    scaler     = MinMaxScaler(feature_range=(0, 1))
    scaled_all = scaler.fit_transform(prices)

    scaled_test = scaled_all[split - LOOKBACK:]
    x_test = np.array([
        scaled_test[i - LOOKBACK:i]
        for i in range(LOOKBACK, len(scaled_test))
    ])

    preds     = model.predict(x_test, verbose=0)
    inv_preds = scaler.inverse_transform(preds).reshape(-1)
    actual    = prices[split:].reshape(-1)
    test_idx  = df.index[split:]

    rmse       = float(np.sqrt(np.mean((inv_preds - actual) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((actual[:-1] - actual[1:]) ** 2)))

    return test_idx, actual, inv_preds, rmse, naive_rmse, split

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# CHANGE: in-chart Plotly legend removed entirely (showlegend=False) — it was
# overlapping the y-axis $ labels on the left. The colored-dot caption below
# each chart already explains what each line is, so the legend was redundant.
# Left margin reclaimed since it's no longer needed to make room for one.
# Plotly's own pan/zoom/etc. toolbar (the "modebar") still renders top-right
# by default — config=dict(displayModeBar=True) just keeps it always visible.
# ══════════════════════════════════════════════════════════════════════════════
BASE_LAYOUT = dict(
    height=360,
    margin=dict(l=10, r=10, t=10, b=0),
    plot_bgcolor="#fafbfc",
    paper_bgcolor="white",
    showlegend=False,
    xaxis=dict(showgrid=True, gridcolor="#eff0f3", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#eff0f3", zeroline=False, tickprefix="$"),
    hovermode="x unified",
)

CHART_CONFIG = dict(displayModeBar=True)  # keeps the pan/zoom/etc. toolbar visible, top-right

def history_chart(df, show_ma100, show_ma250):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Adj Close"],
        name="Price", line=dict(color="#2563eb", width=1.8)
    ))
    if show_ma100:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA100"],
            name="MA 100", line=dict(color="#f59e0b", width=1.3, dash="dot")
        ))
    if show_ma250:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA250"],
            name="MA 250", line=dict(color="#10b981", width=1.3, dash="dash")
        ))
    fig.update_layout(**BASE_LAYOUT)
    return fig

def prediction_chart(test_idx, actual, predicted):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_idx, y=actual,
        name="Actual", line=dict(color="#2563eb", width=1.8)
    ))
    fig.add_trace(go.Scatter(
        x=test_idx, y=predicted,
        name="Predicted", line=dict(color="#ef4444", width=1.5, dash="dash")
    ))
    fig.update_layout(**BASE_LAYOUT)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>📈 FAANG Stock Predictor</h1>
    <p>LSTM neural network &nbsp;·&nbsp; 20-year history &nbsp;·&nbsp; Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)

selected = st.radio(
    "Select company",
    options=list(TICKERS.keys()),
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown(
    f"<div class='info-box'><strong>{TICKERS[selected]}</strong> — {DESCRIPTIONS[selected]}</div>",
    unsafe_allow_html=True,
)

with st.spinner(f"Fetching {selected} data…"):
    data = load_data(selected)

with st.spinner(f"Loading {selected} model…"):
    model = load_model_file(selected)

tab1, tab2, tab3 = st.tabs(["📊  Price History", "🤖  Predictions", "ℹ️  About"])

# ── Tab 1: Price History ────────────────────────────────────────────────────
with tab1:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

    with ctrl_col1:
        years = st.slider("Years of history", min_value=1, max_value=20, value=5, step=1)

    with ctrl_col2:
        show_ma100 = st.checkbox(
            "MA 100", value=True,
            help="100-day moving average — the average closing price over the last 100 trading days. Smooths out short-term noise."
        )

    with ctrl_col3:
        show_ma250 = st.checkbox(
            "MA 250", value=True,
            help="250-day moving average — roughly one trading year. Shows the longer-term trend, smoothing out even more short-term swings."
        )

    cutoff  = datetime.now().year - years
    view_df = data[data.index.year >= cutoff].copy()
    view_df["MA100"] = view_df["Adj Close"].rolling(100).mean()
    view_df["MA250"] = view_df["Adj Close"].rolling(250).mean()

    st.plotly_chart(
        history_chart(view_df, show_ma100, show_ma250),
        use_container_width=True,
        config=CHART_CONFIG,
    )

    # CHANGE: legend explainer — Plotly legends can't show tooltips on hover,
    # and hover doesn't exist on touch devices at all, so this caption
    # explains each line visibly and identically on every device instead.
    st.markdown(
        "<p class='footnote'>"
        "🔵 <strong>Price</strong> — daily closing price&nbsp;&nbsp;·&nbsp;&nbsp;"
        "🟠 <strong>MA 100</strong> — 100-day moving average&nbsp;&nbsp;·&nbsp;&nbsp;"
        "🟢 <strong>MA 250</strong> — 250-day moving average"
        "</p>",
        unsafe_allow_html=True,
    )

    hi52 = float(data["Adj Close"].tail(252).max())
    lo52 = float(data["Adj Close"].tail(252).min())
    s1, s2, s3 = st.columns(3)

    latest_close = float(data["Adj Close"].iloc[-1])
    prev_close   = float(data["Adj Close"].iloc[-2])
    day_delta    = latest_close - prev_close
    day_pct      = day_delta / prev_close * 100

    s1.metric("Latest close",  f"${latest_close:.2f}", f"{day_delta:+.2f} ({day_pct:+.1f}%)")
    s2.metric("52-week high",  f"${hi52:.2f}")
    s3.metric("52-week low",   f"${lo52:.2f}")

# ── Tab 2: Predictions ──────────────────────────────────────────────────────
with tab2:
    with st.spinner("Running predictions…"):
        test_idx, actual, predicted, rmse, naive_rmse, split = run_predictions(data, model)

    st.plotly_chart(
        prediction_chart(test_idx, actual, predicted),
        use_container_width=True,
        config=CHART_CONFIG,
    )

    st.markdown(
        "<p class='footnote'>"
        "🔵 <strong>Actual</strong> — real closing price&nbsp;&nbsp;·&nbsp;&nbsp;"
        "🔴 <strong>Predicted</strong> — the LSTM model's forecast for that day"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Key numbers")

    test_samples = len(data) - split
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("LSTM RMSE",      f"${rmse:.2f}")
    m2.metric("Naive baseline", f"${naive_rmse:.2f}",
              help="Naive = predicting tomorrow = today")
    m3.metric("LSTM % error",   f"{rmse / latest_close * 100:.1f}%")
    m4.metric("Test samples",   f"{test_samples:,}")

    st.markdown(
        "<p class='footnote'>"
        "RMSE (Root Mean Squared Error) is in USD — lower is better. "
        "The naive baseline simply predicts that tomorrow's price equals today's. "
        "% error = RMSE ÷ latest closing price."
        "</p>",
        unsafe_allow_html=True,
    )

# ── Tab 3: About ────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    #### What does this app do?
    This app uses a deep learning model called an **LSTM (Long Short-Term Memory)**
    network to predict stock closing prices based on the past 100 trading days of history.
    It was trained separately on 20 years of data for each FAANG company.

    #### What is RMSE?
    **Root Mean Squared Error** measures how far off predictions are, in dollars.
    An RMSE of $8 means the model's predictions are off by $8 on average.
    Lower is better.

    #### What is the naive baseline?
    The simplest possible prediction: *tomorrow's price = today's price.*
    It's a useful benchmark — if the LSTM can't beat it, the model isn't adding value.

    #### Data source
    All price data is fetched live from **Yahoo Finance** via `yfinance`.
    Data covers the last 20 years from today's date.

    #### Disclaimer
    This app is for **educational purposes only** and should not be used
    for real financial decisions. Stock prices are inherently unpredictable.
    """)