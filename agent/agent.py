from __future__ import annotations
import json, os, datetime as dt
from typing import Dict, Any, List
import yfinance as yf

from .strategy import plan_rebalance, get_targets
from .broker import execute_orders

STATE_PATH = os.path.join("agent", "portfolio_state.json")

def _today_str() -> str:
    return dt.date.today().isoformat()

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "risk_profile": "medium",
        "cash": 0.0,
        "positions": {},        # {ticker: shares}
        "last_run": None,       # "YYYY-MM-DD"
        "history": []           # list of fills/notes
    }

def save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def fetch_prices(tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    # Use 5d to avoid intraday API quirks and still get a last Close
    data = yf.download(tickers, period="5d", auto_adjust=False, progress=False)
    closes = {}
    if isinstance(tickers, str):
        tickers = [tickers]
    try:
        if isinstance(data.columns, yf.pdr_multiindex.MultiIndex) or hasattr(data.columns, "levels"):
            # MultiIndex (multiple tickers)
            last_row = data["Close"].ffill().iloc[-1]
            for t in tickers:
                closes[t] = float(last_row.get(t, float("nan")))
        else:
            # Single ticker
            closes[tickers[0]] = float(data["Close"].ffill().iloc[-1])
    except Exception:
        # best-effort fallback
        for t in tickers:
            try:
                info = yf.Ticker(t).history(period="5d")["Close"].ffill().iloc[-1]
                closes[t] = float(info)
            except Exception:
                closes[t] = 0.0
    # filter zeros
    return {k: v for k, v in closes.items() if v and v > 0}

def portfolio_value(state: Dict[str, Any], prices: Dict[str, float]) -> float:
    total = float(state.get("cash", 0.0))
    for t, q in state.get("positions", {}).items():
        p = float(prices.get(t, 0.0))
        total += float(q) * p
    return round(total, 2)

def run_once(state: Dict[str, Any]) -> Dict[str, Any]:
    risk = state["risk_profile"]
    targets = get_targets(risk)
    tickers = list(targets.keys())

    prices = fetch_prices(tickers)
    if not prices:
        state["history"].append({"ts": _today_str(), "event": "no_prices"})
        return state

    # Plan and execute trades
    orders = plan_rebalance(
        risk=risk,
        cash=state["cash"],
        positions=state["positions"],
        prices=prices
    )
    state["cash"], state["positions"], fills = execute_orders(state["cash"], state["positions"], prices, orders)
    for f in fills:
        state["history"].append({
            "ts": f.ts, "ticker": f.ticker, "side": f.side, "qty": f.qty, "price": f.price, "dollars": f.ticker, "note": f.note
        })

    state["last_run"] = _today_str()
    return state

def maybe_auto_run(state: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    # run at most once per day automatically
    if force or state.get("last_run") != _today_str():
        state = run_once(state)
        save_state(state)
    return state
