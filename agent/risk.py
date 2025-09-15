from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List

def _download_adj_close(tickers: List[str], start, end) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    # yfinance returns different shapes for 1 vs many tickers
    if isinstance(data.columns, pd.MultiIndex):
        adj = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]
    else:
        # single ticker
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        adj = data[[col]].rename(columns={col: tickers[0]})
    return adj.dropna(how="all")

def compute_risk_metrics(tickers: List[str], start, end, min_obs: int = 120) -> pd.DataFrame:
    """Return DataFrame: ticker, vol (ann.), mean_ret (ann.), risk bucket."""
    adj = _download_adj_close(tickers, start, end)
    if adj.empty:
        return pd.DataFrame(columns=["ticker", "vol", "mean_ret", "risk"])

    if isinstance(adj, pd.Series):
        adj = adj.to_frame()

    # daily returns
    rets = adj.pct_change().dropna(how="all")
    # drop columns with too few observations
    good = [c for c in rets.columns if rets[c].dropna().shape[0] >= min_obs]
    rets = rets[good]
    if rets.empty:
        return pd.DataFrame(columns=["ticker", "vol", "mean_ret", "risk"])

    vol = rets.std(skipna=True) * np.sqrt(252)
    mean = rets.mean(skipna=True) * 252
    df = pd.DataFrame({"ticker": vol.index, "vol": vol.values, "mean_ret": mean.values})

    # dynamic buckets by quantiles
    if len(df) >= 3:
        q1, q2 = df["vol"].quantile([0.33, 0.66])
        def bucket(v):
            if v <= q1: return "low"
            if v >= q2: return "high"
            return "medium"
        df["risk"] = df["vol"].apply(bucket)
    else:
        # fallback: all medium
        df["risk"] = "medium"
    return df.sort_values("vol")
