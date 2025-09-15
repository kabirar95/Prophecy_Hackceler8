from __future__ import annotations
import pandas as pd
import yfinance as yf
from prophet import Prophet
import streamlit as st

@st.cache_data(show_spinner=False)
def _dl_daily(ticker: str, start, end) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    out = df.reset_index().rename(columns={"Date": "ds", "Adj Close": "y"})
    if "y" not in out:
        out = out.rename(columns={"Close": "y"})
    out = out[["ds", "y"]].dropna().sort_values("ds")
    return out

@st.cache_data(show_spinner=False)
def expected_return_prophet(ticker: str, start, end, horizon_days: int = 30) -> float:
    """
    Fit a small Prophet on daily closes and return expected % change over horizon.
    Safe: returns 0.0 if fit fails or not enough data.
    """
    try:
        df = _dl_daily(ticker, start, end)
        if df.shape[0] < 60:
            return 0.0
        m = Prophet(daily_seasonality=True)
        m.fit(df.rename(columns={"ds": "ds", "y": "y"}))
        future = m.make_future_dataframe(periods=horizon_days, freq="B")
        fc = m.predict(future)
        last = df["y"].iloc[-1]
        fut = fc["yhat"].iloc[-1]
        if last <= 0:
            return 0.0
        return float((fut - last) / last)
    except Exception:
        return 0.0
