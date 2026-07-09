# Instant Multi-Asset Portfolio VaR

Drop a price file, get Parametric, Historical, and Monte Carlo VaR — plus
component VaR and backtesting — in under a second, no matter how large the
underlying dataset. Built to replace an Excel workbook where formulas were
timing out on ~246MB of data.

## Files

- `var_engine.py` — the actual math (no UI dependency, importable/testable on its own)
- `app.py` — Streamlit interface: upload, weights, results
- `requirements.txt` — dependencies

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`. Check "use synthetic demo
data" first to confirm everything runs, then upload your real file.

## Handling your 246MB file specifically

**Expected input shape:** a wide table — first column = date, every other
column = one asset's price series (not returns; the app computes returns
for you).

**The first load will still take a little while** because `pd.read_excel`
on a 246MB `.xlsx` is inherently slow (xlsx is XML under the hood). What
matters is that this only happens **once**:

1. On first upload, the app parses the Excel file and immediately writes a
   cached copy to `.var_cache/<filename>.parquet`.
2. Every subsequent run — different weights, different confidence level,
   re-opening the app tomorrow — reads that Parquet file instead, which
   loads in a couple of seconds even at millions of rows.

**If the first load is too slow or the file won't upload through the
browser widget at all** (Streamlit's default upload cap is 200MB), do the
conversion once ahead of time instead:

```python
import pandas as pd
df = pd.read_excel("your_246mb_file.xlsx", index_col=0, engine="openpyxl")
df.to_parquet("your_data.parquet")
```

Then upload the `.parquet` file to the app instead of the original xlsx —
the app supports it natively and skips the Excel parser entirely.

If you'd rather raise Streamlit's upload limit instead of pre-converting,
add this to `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 500
```

## What "weights" means here

- **Equal weight**: quick sanity check across all assets equally.
- **Manual entry**: enter actual position sizes or target weights per
  ticker — the app normalizes them to sum to 1 automatically, so you can
  enter raw ₹ amounts instead of percentages if that's easier.

## Reading the output

- **Three VaR numbers side by side** — if they're close, your data is
  well-behaved (roughly normal). If Historical/Monte Carlo run noticeably
  higher than Parametric, your portfolio has fat tails the normal-distribution
  assumption misses — trust the empirical methods more in that case.
- **Component VaR** — breaks the total portfolio VaR down by holding, so
  you can see which position is actually driving risk (useful even when a
  position is small in ₹ terms but highly correlated with everything else).
- **Backtest** — checks how often actual historical losses exceeded the VaR
  estimate. For 95% VaR, expect ~5% of days to breach; a much higher rate
  means the model is understating risk, a much lower rate means it's overly
  conservative.

## Extending this

- Swap in a GARCH volatility model instead of the flat historical
  covariance matrix if you want VaR to react to recent volatility regime
  changes rather than a full-sample average.
- Add expected shortfall (CVaR) — the average loss *beyond* the VaR
  threshold — it's one extra line off the historical/Monte Carlo series and
  is increasingly preferred over VaR alone for tail-risk reporting.
- If this needs to run unattended (e.g., a daily risk report emailed out),
  strip the Streamlit UI and call the functions in `var_engine.py` directly
  from a scheduled script, writing results to Excel via `openpyxl` for
  distribution.
