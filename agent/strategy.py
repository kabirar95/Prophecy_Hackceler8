from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

# Drift thresholds: how far we'll allow allocations to deviate from target before rebalancing
REBALANCE_THRESHOLDS = {
    "low": 0.05,     # ±5%
    "medium": 0.10,  # ±10%
    "high": 0.25     # ±25%
}

# Simple, hackathon-friendly universes per risk profile.
# (Later we can swap this for a momentum/top-N picker without changing the agent interface.)
UNIVERSES = {
    "low":   ["VOO", "MSFT", "JNJ", "BND"],
    "medium":["VOO", "MSFT", "AAPL", "AMZN", "TSLA"],
    "high":  ["TSLA", "NVDA", "AAPL", "AMZN", "META"]
}

# Target weights per risk profile (must sum ~1.0)
TARGETS = {
    "low":   {"VOO": 0.60, "MSFT": 0.15, "JNJ": 0.10, "BND": 0.15},
    "medium":{"VOO": 0.40, "MSFT": 0.20, "AAPL": 0.15, "AMZN": 0.10, "TSLA": 0.15},
    "high":  {"TSLA": 0.35, "NVDA": 0.25, "AAPL": 0.15, "AMZN": 0.10, "META": 0.15}
}

@dataclass
class Order:
    ticker: str
    side: str     # "buy" or "sell"
    qty: int
    note: str = ""

def get_targets(risk: str) -> Dict[str, float]:
    risk = (risk or "medium").lower()
    if risk not in TARGETS:
        risk = "medium"
    return TARGETS[risk]

def get_threshold(risk: str) -> float:
    risk = (risk or "medium").lower()
    return REBALANCE_THRESHOLDS.get(risk, 0.10)

def current_allocations(positions: Dict[str, int], prices: Dict[str, float]) -> Dict[str, float]:
    value = 0.0
    for tkr, q in positions.items():
        px = prices.get(tkr, 0.0)
        value += max(q, 0) * max(px, 0.0)
    if value <= 0:
        return {t: 0.0 for t in prices.keys()}
    return {t: (max(positions.get(t, 0), 0) * max(prices.get(t, 0.0), 0.0)) / value for t in prices.keys()}

def _round_qty(dollars: float, price: float) -> int:
    if price <= 0:
        return 0
    # whole-share round, ignore fractional shares for now
    return max(int(dollars // price), 0)

def plan_rebalance(
    risk: str,
    cash: float,
    positions: Dict[str, int],
    prices: Dict[str, float],
    min_trade_dollars: float = 50.0
) -> List[Order]:
    """
    Generate market orders to move allocations toward target if drift > threshold.
    High risk -> larger tolerance (let winners run). Low risk -> tighter control.
    """
    targets = get_targets(risk)
    threshold = get_threshold(risk)

    # Restrict to tickers we actually have prices for
    tickers = [t for t in targets.keys() if t in prices]
    if not tickers:
        return []

    # Compute current portfolio value
    port_value = cash + sum(max(positions.get(t, 0), 0) * max(prices.get(t, 0.0), 0.0) for t in tickers)
    if port_value <= 0:
        # fresh start: allocate cash to targets
        orders: List[Order] = []
        for t in tickers:
            target_dollars = targets[t] * cash
            qty = _round_qty(target_dollars, prices[t])
            if qty > 0:
                orders.append(Order(ticker=t, side="buy", qty=qty, note="initial allocation"))
        return orders

    # Where we are vs where we want to be
    alloc_now = current_allocations(positions, {t: prices[t] for t in tickers})

    # Decide sells first (if overweight beyond threshold)
    orders: List[Order] = []
    for t in tickers:
        diff = alloc_now.get(t, 0.0) - targets[t]
        if diff > threshold:
            # sell down to edge of band (target + threshold/2) to reduce churn
            desired_alloc = max(targets[t] + threshold * 0.5, targets[t])
            desired_dollars = desired_alloc * port_value
            have_dollars = max(positions.get(t, 0), 0) * prices[t]
            delta = have_dollars - desired_dollars
            if delta > min_trade_dollars:
                qty = _round_qty(delta, prices[t])
                if qty > 0:
                    orders.append(Order(ticker=t, side="sell", qty=qty, note="rebalance trim"))

    # Recompute cash after hypothetical sells
    sim_cash = cash
    sim_positions = positions.copy()
    for o in orders:
        if o.side == "sell":
            px = prices[o.ticker]
            held = sim_positions.get(o.ticker, 0)
            sell_qty = min(o.qty, max(held, 0))
            sim_positions[o.ticker] = max(held - sell_qty, 0)
            sim_cash += sell_qty * px

    # Buy underweights next
    port_value_post = sim_cash + sum(max(sim_positions.get(t, 0), 0) * prices[t] for t in tickers)
    alloc_after_sells = current_allocations(sim_positions, {t: prices[t] for t in tickers})

    # Simple pass to push underweights toward target within tolerance
    for t in tickers:
        diff = targets[t] - alloc_after_sells.get(t, 0.0)
        if diff > threshold:
            desired_alloc = max(targets[t] - threshold * 0.5, 0.0)
            desired_dollars = desired_alloc * port_value_post
            have_dollars = max(sim_positions.get(t, 0), 0) * prices[t]
            delta = desired_dollars - have_dollars
            if delta > min_trade_dollars and sim_cash > prices[t]:
                qty = _round_qty(min(delta, sim_cash), prices[t])
                if qty > 0:
                    orders.append(Order(ticker=t, side="buy", qty=qty, note="rebalance add"))
                    sim_cash -= qty * prices[t]

    return orders
