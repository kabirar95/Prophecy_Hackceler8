from typing import List

# Fallback universe (popular, liquid). User can input any list at runtime.
DEFAULT_UNIVERSE = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","AMD","NFLX","AVGO",
    "BRK-B","JPM","V","MA","KO","PEP","DIS","INTC","CSCO","ADBE",
    "XOM","CVX","PFE","JNJ","WMT","HD","PG","NKE","BAC","TSM",
    "SHOP","CRM","LIN","MRK","ABBV","UNH","ORCL","QCOM","TXN","AMAT"
]

def parse_user_universe(text: str) -> List[str]:
    if not text:
        return []
    raw = [t.strip().upper().replace(" ", "") for t in text.split(",")]
    # De-duplicate & keep only non-empty tickers
    seen = set()
    out = []
    for t in raw:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out
