"""
Target-allocation-drift rebalance calculator.

Given current positions, live prices, and target weights, this computes
which trades would bring the portfolio back to target -- flagging only
positions that have drifted beyond a threshold, rather than rebalancing
everything to the decimal every time (which would be noisy and unrealistic).

This is deliberately simple on purpose: no Sharpe ratio, no mean-variance
optimization. Get this loop (real data -> compute -> proposed trade ->
approval) working end to end first. Fancier math is a stretch goal, not
a prerequisite.
"""

from __future__ import annotations


def compute_rebalance(
    positions: list[dict],
    prices: dict[str, float],
    target_weights: dict[str, float],
    cash: float = 0.0,
    drift_threshold: float = 0.05,
) -> dict:
    """
    positions: [{"symbol": "AAPL", "qty": 10}, ...]
    prices: {"AAPL": 227.50, ...}  -- current price per share
    target_weights: {"AAPL": 0.3, "MSFT": 0.3, "SPY": 0.4}  -- must sum to <= 1.0
    cash: uninvested cash, counts toward total portfolio value
    drift_threshold: only propose a trade if current weight is off by more
                      than this (e.g. 0.05 = 5 percentage points)
    """
    current_values = {p["symbol"]: p["qty"] * prices[p["symbol"]] for p in positions}
    total_value = cash + sum(current_values.values())

    if total_value <= 0:
        return {"total_value": 0.0, "trades": [], "note": "No portfolio value to rebalance."}

    all_symbols = sorted(set(target_weights) | set(current_values))
    trades = []

    for symbol in all_symbols:
        current_value = current_values.get(symbol, 0.0)
        current_weight = current_value / total_value
        target_weight = target_weights.get(symbol, 0.0)
        drift = current_weight - target_weight

        if abs(drift) < drift_threshold:
            continue

        price = prices.get(symbol)
        if price is None or price <= 0:
            # Can't size a trade without a price -- flag it instead of guessing.
            trades.append({
                "symbol": symbol,
                "action": "skipped_no_price",
                "current_weight": round(current_weight, 4),
                "target_weight": round(target_weight, 4),
                "drift": round(drift, 4),
            })
            continue

        target_value = target_weight * total_value
        trade_value = target_value - current_value
        trade_qty = trade_value / price

        trades.append({
            "symbol": symbol,
            "action": "buy" if trade_qty > 0 else "sell",
            "qty": round(abs(trade_qty), 4),
            "est_value": round(abs(trade_value), 2),
            "current_weight": round(current_weight, 4),
            "target_weight": round(target_weight, 4),
            "drift": round(drift, 4),
        })

    return {"total_value": round(total_value, 2), "trades": trades}


if __name__ == "__main__":
    import json

    # Sample data shaped like what Alpaca's get_positions / get_stock_quote
    # tools would actually return -- swap this for real tool output later.
    sample_positions = [
        {"symbol": "AAPL", "qty": 100},
        {"symbol": "MSFT", "qty": 50},
        {"symbol": "SPY", "qty": 20},
    ]
    sample_prices = {"AAPL": 227.50, "MSFT": 415.20, "SPY": 560.10}
    sample_targets = {"AAPL": 0.30, "MSFT": 0.30, "SPY": 0.40}
    sample_cash = 5000.0

    result = compute_rebalance(
        positions=sample_positions,
        prices=sample_prices,
        target_weights=sample_targets,
        cash=sample_cash,
        drift_threshold=0.05,
    )
    print(json.dumps(result, indent=2))
