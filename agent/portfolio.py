from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def propose_target_weights(risk_level: str, metrics_df: pd.DataFrame, exp_returns: Dict[str, float], max_positions=8):
    """Return dict ticker->weight for chosen risk bucket using exp_return/vol score."""
    if metrics_df.empty:
        return {}
    df = metrics_df.copy()
    df["exp_return"] = df["ticker"].map(exp_returns).fillna(0.0)
    pool = df[df["risk"] == risk_level].copy()
    pool = pool[pool["exp_return"] > 0]  # require positive expectation
    if pool.empty:
        # fallback: take the lowest vol names
        pool = df.sort_values("vol").head(max_positions).copy()
        pool["exp_return"] = pool["exp_return"].clip(lower=0.0)
    pool["score"] = pool["exp_return"] / (pool["vol"].replace(0, np.nan))
    pool["score"] = pool["score"].replace([np.inf, -np.inf, np.nan], 0.0)
    top = pool.sort_values("score", ascending=False).head(max_positions).copy()
    if top.empty:
        return {}
    # inverse-vol * positive exp return
    raw = (1.0 / (top["vol"].replace(0, np.nan))) * top["exp_return"].clip(lower=0.0)
    raw = raw.replace([np.inf, -np.inf, np.nan], 0.0)
    if raw.sum() <= 0:
        # equal weight fallback
        w = np.repeat(1.0 / len(top), len(top))
    else:
        w = raw / raw.sum()
    return dict(zip(top["ticker"], w))

def generate_rebalance_orders(current_positions: dict, cash: float, prices: dict, target_weights: dict):
    """
    Return list of orders: {ticker, side, shares, est_value}. Allows fractional shares.
    """
    _tickers = sorted(set(list(current_positions.keys()) + list(target_weights.keys())))
    equity = cash + sum(prices.get(t, 0.0) * current_positions.get(t, 0.0) for t in _tickers)
    orders = []
    for t in _tickers:
        px = float(prices.get(t, 0.0))
        cur_sh = float(current_positions.get(t, 0.0))
        tgt_w = float(target_weights.get(t, 0.0))
        tgt_val = equity * tgt_w
        cur_val = cur_sh * px
        diff_val = tgt_val - cur_val
        if px <= 0:
            continue
        shares = diff_val / px
        if abs(shares) < 1e-6:
            continue
        side = "BUY" if shares > 0 else "SELL"
        orders.append({"ticker": t, "side": side, "shares": float(shares), "est_value": float(abs(shares) * px)})
    return orders
