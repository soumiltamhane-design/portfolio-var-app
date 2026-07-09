"""
Instant Multi-Asset VaR — drop a file, get VaR in seconds.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Handles the exact problem this was built for: a huge Excel file that chokes
native Excel formulas. Large .xlsx files are converted once to Parquet
(cached on disk) so every subsequent run/recompute is instant.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from var_engine import (
    prices_to_returns, normalize_weights,
    parametric_var, historical_var, monte_carlo_var,
    component_var, backtest_var,
)

st.set_page_config(page_title="Instant Portfolio VaR", layout="wide")

CACHE_DIR = Path(".var_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# File loading with Parquet caching (this is what kills the 246MB problem)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_prices(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Loads a wide dates x assets price file. Expects first column = date.
    Caches a Parquet copy so re-runs (or re-computes with different weights)
    skip the expensive Excel parse entirely.
    """
    suffix = Path(filename).suffix.lower()
    cache_path = CACHE_DIR / f"{filename}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    tmp_path = CACHE_DIR / f"_raw_{filename}"
    tmp_path.write_bytes(file_bytes)

    t0 = time.time()
    if suffix in (".xlsx", ".xlsm"):
        df = pd.read_excel(tmp_path, index_col=0, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(tmp_path, index_col=0)
    elif suffix == ".parquet":
        df = pd.read_parquet(tmp_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df.apply(pd.to_numeric, errors="coerce")

    df.to_parquet(cache_path)
    tmp_path.unlink(missing_ok=True)

    st.session_state["_last_load_seconds"] = round(time.time() - t0, 1)
    return df


def make_synthetic_demo(n_days=1000, n_assets=8, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"STOCK_{i+1}" for i in range(n_assets)]
    market = rng.normal(0.0005, 0.01, n_days)
    betas = rng.uniform(0.4, 1.6, n_assets)
    idio = rng.normal(0, 0.015, (n_days, n_assets))
    daily_returns = market[:, None] * betas[None, :] + idio
    prices = 100 * np.cumprod(1 + daily_returns, axis=0)
    return pd.DataFrame(prices, columns=tickers,
                         index=pd.date_range("2022-01-01", periods=n_days, freq="B"))


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

st.title("📉 Instant Multi-Asset Portfolio VaR")
st.caption("Drop a price file → get Parametric, Historical & Monte Carlo VaR in seconds — no Excel recalculation.")

with st.sidebar:
    st.header("1. Data")
    use_demo = st.checkbox("Use synthetic demo data instead of uploading", value=True)

    uploaded = None
    if not use_demo:
        uploaded = st.file_uploader(
            "Upload price file (dates x assets, wide format)",
            type=["xlsx", "csv", "parquet"],
        )
        st.caption(
            "First column = date, remaining columns = one price series per asset. "
            "Large .xlsx files are cached to Parquet after the first load — "
            "every recompute after that is instant."
        )

    st.header("2. VaR settings")
    confidence = st.select_slider("Confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
    horizon_days = st.number_input("Horizon (days)", min_value=1, max_value=30, value=1)
    portfolio_value = st.number_input("Portfolio value (₹)", min_value=0.0, value=10_000_000.0, step=100000.0, format="%.0f")
    n_sims = st.number_input("Monte Carlo simulations", min_value=1000, max_value=100000, value=20000, step=1000)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

if use_demo:
    prices = make_synthetic_demo()
    st.info("Using synthetic demo data (8 correlated assets, 1000 trading days). Uncheck the box in the sidebar to upload your own file.")
elif uploaded is not None:
    prices = load_prices(uploaded.getvalue(), uploaded.name)
    load_time = st.session_state.get("_last_load_seconds")
    if load_time:
        st.success(f"Loaded and cached to Parquet in {load_time}s. Future recomputes on this file will be instant.")
else:
    st.warning("Upload a file or check 'use synthetic demo data' to continue.")
    st.stop()

returns = prices_to_returns(prices)
tickers = list(returns.columns)

st.subheader("Loaded data")
c1, c2, c3 = st.columns(3)
c1.metric("Assets", len(tickers))
c2.metric("Trading days", len(returns))
c3.metric("Date range", f"{returns.index.min().date()} → {returns.index.max().date()}")

with st.expander("Preview returns matrix"):
    st.dataframe(returns.tail(10), use_container_width=True)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

st.subheader("3. Portfolio weights")
weight_mode = st.radio("Set weights", ["Equal weight", "Manual entry"], horizontal=True)

if weight_mode == "Equal weight":
    raw_weights = {t: 1.0 for t in tickers}
else:
    default_df = pd.DataFrame({"ticker": tickers, "weight": [1.0] * len(tickers)})
    edited = st.data_editor(default_df, hide_index=True, use_container_width=True, key="weights_editor")
    raw_weights = dict(zip(edited["ticker"], edited["weight"]))

weights = normalize_weights(raw_weights)

if not st.button("🚀 Compute VaR", type="primary"):
    st.stop()


# ---------------------------------------------------------------------------
# Compute — this whole block should run in well under a second
# ---------------------------------------------------------------------------

t0 = time.time()

p_res, cov_matrix, marginal_var = parametric_var(returns, weights, portfolio_value, confidence, horizon_days)
h_res = historical_var(returns, weights, portfolio_value, confidence, horizon_days)
mc_res = monte_carlo_var(returns, weights, portfolio_value, confidence, horizon_days, n_sims=n_sims)

w_aligned = weights.reindex(returns.columns).fillna(0.0)
port_vol_daily = float(np.sqrt(w_aligned.values @ cov_matrix.values @ w_aligned.values.T))
comp = component_var(marginal_var, weights, portfolio_value, p_res.var_pct, port_vol_daily)

bt = backtest_var(returns, weights, p_res.var_pct)

compute_time = time.time() - t0

st.success(f"Computed all three VaR methods + component breakdown + backtest in {compute_time:.3f} seconds.")

st.subheader(f"4. Results — {int(confidence*100)}% confidence, {horizon_days}-day horizon")

col1, col2, col3 = st.columns(3)
for col, res in zip([col1, col2, col3], [p_res, h_res, mc_res]):
    col.metric(res.method, f"{res.var_pct*100:.2f}%", f"₹ {res.var_amount:,.0f}")

st.caption(
    "If the three methods diverge a lot, your return distribution likely has fat tails "
    "or skew that the parametric (normal-distribution) method underestimates — trust "
    "Historical/Monte Carlo more in that case."
)

st.subheader("5. Component VaR — which holdings drive the risk")
st.dataframe(
    comp.style.format({
        "weight": "{:.2%}",
        "component_var_pct": "{:.3%}",
        "component_var_amount": "₹ {:,.0f}",
        "pct_of_total_var": "{:.1%}",
    }),
    use_container_width=True,
)
st.bar_chart(comp["component_var_amount"])

st.subheader("6. Backtest — did actual losses exceed VaR as often as expected?")
expected_breach_rate = 1 - confidence
bcol1, bcol2, bcol3 = st.columns(3)
bcol1.metric("Trading days tested", bt["total_days"])
bcol2.metric("Breaches", bt["breaches"])
bcol3.metric("Actual breach rate", f"{bt['breach_rate']*100:.2f}%", f"expected ≈ {expected_breach_rate*100:.1f}%")

if bt["breach_rate"] > expected_breach_rate * 1.5:
    st.warning("Breach rate is notably higher than expected — the VaR model may be understating risk.")
elif bt["breach_rate"] < expected_breach_rate * 0.5:
    st.info("Breach rate is notably lower than expected — the VaR model may be overly conservative.")
