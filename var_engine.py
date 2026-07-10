"""
Multi-Asset Portfolio VaR Engine
----------------------------------
Computes Value at Risk using three methods (Parametric, Historical, Monte Carlo)
plus Component/Marginal VaR breakdown by position.

Design goal: everything here operates on a returns matrix (dates x assets) and
a weights vector, using vectorized numpy/pandas — no per-cell recalculation,
so this runs in milliseconds even with hundreds of assets and years of daily data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Above this many held assets, a full dense covariance matrix (N x N floats,
# duplicated again inside the Monte Carlo Cholesky step) gets big enough to
# exhaust memory on memory-constrained deployments (e.g. Streamlit Community
# Cloud's free tier, ~1GB RAM). When that happens, the C-level BLAS/LAPACK
# code can crash with a segmentation fault instead of Python raising a clean
# MemoryError — so we guard against it explicitly rather than letting the
# whole process die.
MAX_ASSETS_FOR_FULL_COV = 1500


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def prices_to_returns(prices: pd.DataFrame, min_observations: int = 30) -> tuple[pd.DataFrame, list]:
    """
    Convert a wide dates x assets price matrix into simple daily returns.

    Real portfolios (especially bonds) have assets with wildly different
    trading histories — some start in 2019, some in 2024, some mature early.
    Matrix math (covariance, matrix multiplication) breaks completely the
    moment even one cell is NaN, so this function:

    1. Drops any asset with fewer than `min_observations` valid price points
       (not enough history to estimate its risk contribution reliably).
    2. Fills any remaining isolated gaps with 0 return for that day — this
       only happens for scattered single-day gaps that survive upstream
       forward-filling; it does NOT restore assets that were dropped in step 1.

    Returns (returns_df, dropped_asset_names) so the caller can report what
    was excluded and why.
    """
    prices = prices.sort_index()

    valid_counts = prices.notna().sum()
    dropped = list(valid_counts[valid_counts < min_observations].index)
    prices = prices[[c for c in prices.columns if c not in dropped]]

    returns = prices.pct_change().dropna(how="all")
    returns = returns.dropna(axis=1, how="all")

    # any stray inf (e.g. a data glitch that slipped through as a genuine
    # divide-by-zero) is treated the same as a missing observation
    returns = returns.replace([np.inf, -np.inf], np.nan)

    # any remaining scattered gaps (e.g. an asset's first day, one-off missing
    # print) become 0 — a deliberate simplification once low-history assets
    # are already excluded, so covariance/matrix math never hits a NaN.
    returns = returns.fillna(0.0)

    return returns, dropped


def normalize_weights(weights: dict) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    if w.sum() == 0:
        raise ValueError("Weights sum to zero — check position sizes.")
    return w / w.sum()


# ---------------------------------------------------------------------------
# Core VaR methods
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    method: str
    confidence: float
    horizon_days: int
    var_pct: float          # as a positive fraction, e.g. 0.023 = 2.3% loss
    var_amount: float       # in portfolio currency units


def parametric_var(returns: pd.DataFrame, weights: pd.Series, portfolio_value: float,
                    confidence: float = 0.95, horizon_days: int = 1) -> tuple[VaRResult, pd.Series, pd.Series]:
    """
    Variance-covariance method. Assumes returns are approximately normal.
    Returns the VaR result plus the covariance matrix and marginal VaR series
    (used later for component VaR).
    """
    _check_asset_count(returns.shape[1])

    # float32 halves the memory footprint of the covariance matrix and every
    # downstream matrix op — the single biggest lever for avoiding OOM-driven
    # segfaults on large held-security universes. Precision loss is
    # negligible for VaR purposes (well below basis-point level).
    w = weights.reindex(returns.columns).fillna(0.0).values.astype(np.float32)
    cov_matrix = returns.cov().astype(np.float32)
    portfolio_var_daily = w @ cov_matrix.values @ w.T
    portfolio_vol_daily = np.sqrt(max(portfolio_var_daily, 0))
    portfolio_mean_daily = float(returns.mean().values.astype(np.float32) @ w)

    z = _z_score(confidence)
    scale = np.sqrt(horizon_days)

    var_pct = -(portfolio_mean_daily * horizon_days - z * portfolio_vol_daily * scale)
    var_pct = max(var_pct, 0.0)

    result = VaRResult(
        method="Parametric (Variance-Covariance)",
        confidence=confidence,
        horizon_days=horizon_days,
        var_pct=var_pct,
        var_amount=var_pct * portfolio_value,
    )

    marginal_var = (cov_matrix.values @ w) / portfolio_vol_daily if portfolio_vol_daily > 0 else np.zeros_like(w)
    marginal_var = pd.Series(marginal_var, index=returns.columns)

    return result, cov_matrix, marginal_var


def historical_var(returns: pd.DataFrame, weights: pd.Series, portfolio_value: float,
                    confidence: float = 0.95, horizon_days: int = 1) -> VaRResult:
    """
    Historical simulation. No distributional assumption — uses the actual
    empirical distribution of portfolio returns. More robust to fat tails/skew.
    """
    w = weights.reindex(returns.columns).fillna(0.0).values
    portfolio_returns = returns.values @ w  # collapse to single series

    if horizon_days > 1:
        # simple overlapping-window scaling; for real desks consider block bootstrap instead
        portfolio_returns = portfolio_returns * np.sqrt(horizon_days)

    percentile = (1 - confidence) * 100
    var_pct = -np.nanpercentile(portfolio_returns, percentile)
    var_pct = max(var_pct, 0.0)

    return VaRResult(
        method="Historical Simulation",
        confidence=confidence,
        horizon_days=horizon_days,
        var_pct=var_pct,
        var_amount=var_pct * portfolio_value,
    )


def monte_carlo_var(returns: pd.DataFrame, weights: pd.Series, portfolio_value: float,
                     confidence: float = 0.95, horizon_days: int = 1,
                     n_sims: int = 20000, seed: int = 42) -> VaRResult:
    """
    Monte Carlo using a Cholesky decomposition of the covariance matrix, so
    simulated shocks respect the actual correlation structure between assets.
    """
    _check_asset_count(returns.shape[1])

    rng = np.random.default_rng(seed)
    # float32 throughout: the shocks matrix (n_sims x n_assets) is the
    # single largest allocation in this whole app, and doubling it for
    # float64 precision buys nothing meaningful for a VaR estimate.
    w = weights.reindex(returns.columns).fillna(0.0).values.astype(np.float32)
    mean = returns.mean().values.astype(np.float32)
    cov = returns.cov().values.astype(np.float32)

    # jitter for numerical stability if the covariance matrix is near-singular
    cov_stable = cov + np.eye(cov.shape[0], dtype=np.float32) * 1e-10
    try:
        L = np.linalg.cholesky(cov_stable)
    except np.linalg.LinAlgError:
        # Matrix isn't positive-definite even with jitter (can happen with a
        # very large number of highly correlated/collinear assets). Fall back
        # to ignoring cross-asset correlation rather than crashing — less
        # accurate, but still directionally useful and always completes.
        L = np.diag(np.sqrt(np.clip(np.diag(cov_stable), 0, None))).astype(np.float32)

    # Run in batches rather than allocating one (n_sims x n_assets) array up
    # front — for a large held-security universe that single allocation can
    # be the tipping point that exhausts memory. Batching keeps peak memory
    # bounded regardless of n_sims.
    batch_size = max(1, min(n_sims, int(5_000_000 // max(len(w), 1))))
    portfolio_return_batches = []
    remaining = n_sims
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        shocks = rng.standard_normal(size=(this_batch, len(w))).astype(np.float32)
        correlated_shocks = shocks @ L.T
        simulated_asset_returns = mean + correlated_shocks
        portfolio_return_batches.append(simulated_asset_returns @ w)
        remaining -= this_batch
    simulated_portfolio_returns = np.concatenate(portfolio_return_batches)

    if horizon_days > 1:
        simulated_portfolio_returns = simulated_portfolio_returns * np.sqrt(horizon_days)

    percentile = (1 - confidence) * 100
    var_pct = -np.percentile(simulated_portfolio_returns, percentile)
    var_pct = max(var_pct, 0.0)

    return VaRResult(
        method="Monte Carlo",
        confidence=confidence,
        horizon_days=horizon_days,
        var_pct=var_pct,
        var_amount=var_pct * portfolio_value,
    )


# ---------------------------------------------------------------------------
# Component / Marginal VaR
# ---------------------------------------------------------------------------

def component_var(marginal_var: pd.Series, weights: pd.Series, portfolio_value: float,
                   parametric_var_pct: float, portfolio_vol_daily: float) -> pd.DataFrame:
    """
    Breaks total parametric VaR down by asset. Component VaRs sum to the
    total portfolio VaR — shows which holdings are driving risk.
    """
    w = weights.reindex(marginal_var.index).fillna(0.0)
    if portfolio_vol_daily == 0:
        contrib_pct = pd.Series(0.0, index=marginal_var.index)
    else:
        contrib_pct = (w * marginal_var) / portfolio_vol_daily  # fraction of total vol

    component_var_pct = contrib_pct * parametric_var_pct
    component_var_amount = component_var_pct * portfolio_value

    out = pd.DataFrame({
        "weight": w,
        "component_var_pct": component_var_pct,
        "component_var_amount": component_var_amount,
    })
    out["pct_of_total_var"] = out["component_var_amount"] / out["component_var_amount"].sum()
    return out.sort_values("component_var_amount", ascending=False)


def individual_asset_var(returns: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """
    Standalone VaR per asset (no portfolio weighting, no cross-asset
    correlation) — used to rank individual securities by their own risk,
    e.g. "which of my currently-held bonds is riskiest on its own".
    """
    z = _z_score(confidence)
    percentile = (1 - confidence) * 100

    hist_var = returns.apply(lambda col: -np.nanpercentile(col.values, percentile))
    param_var = -(returns.mean() - z * returns.std())

    out = pd.DataFrame({
        "historical_var_pct": hist_var.clip(lower=0),
        "parametric_var_pct": param_var.clip(lower=0),
    })
    return out.sort_values("historical_var_pct", ascending=False)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def backtest_var(returns: pd.DataFrame, weights: pd.Series, var_pct: float) -> dict:
    """
    Counts how often actual portfolio losses exceeded the VaR estimate.
    For a 95% VaR, expect ~5% of days to breach. Big deviations flag model risk.
    """
    w = weights.reindex(returns.columns).fillna(0.0).values
    portfolio_returns = returns.values @ w
    breaches = (portfolio_returns < -var_pct).sum()
    total_days = len(portfolio_returns)
    return {
        "total_days": total_days,
        "breaches": int(breaches),
        "breach_rate": breaches / total_days if total_days else 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_asset_count(n_assets: int) -> None:
    """
    Raises a normal, catchable Python exception (with a clear message) if
    the held-asset universe is large enough that a dense N x N covariance
    matrix risks exhausting memory and crashing the whole process with a
    segfault instead. Callers (e.g. the Streamlit app) should catch this and
    show it as a friendly on-screen error rather than letting the server die.
    """
    if n_assets > MAX_ASSETS_FOR_FULL_COV:
        raise MemoryError(
            f"{n_assets} assets is too many for a full covariance-based VaR "
            f"calculation on this server's available memory (limit is "
            f"{MAX_ASSETS_FOR_FULL_COV}). This is what was likely causing the "
            f"deployed app to crash. Options: restrict to a subset of "
            f"securities (e.g. only the largest positions), or run this "
            f"locally / on a machine with more RAM, where this limit doesn't apply."
        )


def _z_score(confidence: float) -> float:
    # Standard normal inverse CDF for common confidence levels, avoids a scipy dependency
    lookup = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263, 0.995: 2.5758}
    if confidence in lookup:
        return lookup[confidence]
    # fallback: linear nearest-neighbor interpolation isn't right for a CDF,
    # so require scipy for anything non-standard
    from scipy.stats import norm
    return float(norm.ppf(confidence))
