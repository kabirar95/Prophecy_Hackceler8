from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass, asdict
import time

@dataclass
class Fill:
    ts: float
    ticker: str
    side: str   # "buy" | "sell"
    qty: int
    price: float
    dollars: float
    note: str = ""

def execute_orders(
    cash: float,
    positions: Dict[str, int],
    prices: Dict[str, float],
    orders: List,
) -> (float, Dict[str, int], List[Fill]):
    """
    Super simple market execution at current prices.
    Refuses orders it cannot afford or doesn't have shares for.
    """
    fills: List[Fill] = []
    now = time.time()

    for o in orders:
        px = prices.get(o.ticker, 0.0)
        if px <= 0:
            continue

        if o.side == "buy":
            cost = o.qty * px
            if cost <= cash and o.qty > 0:
                cash -= cost
                positions[o.ticker] = positions.get(o.ticker, 0) + o.qty
                fills.append(Fill(now, o.ticker, "buy", o.qty, px, -cost, o.note))

        elif o.side == "sell":
            held = positions.get(o.ticker, 0)
            sell_qty = min(held, o.qty)
            if sell_qty > 0:
                proceeds = sell_qty * px
                cash += proceeds
                positions[o.ticker] = held - sell_qty
                fills.append(Fill(now, o.ticker, "sell", sell_qty, px, proceeds, o.note))

    # Clean empty positions
    positions = {k: v for k, v in positions.items() if v > 0}
    return cash, positions, fills
