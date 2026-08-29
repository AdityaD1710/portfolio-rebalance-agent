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

import math


def compute_rebalance(
    positions: list[dict],
    prices: dict[str, float],
    target_weights: dict[str, float],
    cash: float = 0.0,
    drift_threshold: float = 0.05,
) -> dict:
    """
    positions: [{"symbol": "AAPL", "qty": 10}, ...] -- qty may be a string
               (Alpaca's Position schema returns qty as a string) or a number.
    prices: {"AAPL": 227.50, ...}  -- current price per share
    target_weights: {"AAPL": 0.3, "MSFT": 0.3, "SPY": 0.4} -- must sum to <= 1.0,
                     no negative or non-finite weights
    cash: uninvested cash, counts toward total portfolio value
    drift_threshold: only propose a trade if current weight is off by more
                      than this (e.g. 0.05 = 5 percentage points). A position
                      exactly at the threshold is NOT traded.
    """
    if any(not math.isfinite(w) for w in target_weights.values()):
        raise ValueError("target_weights must be finite numbers")
    if any(w < 0 for w in target_weights.values()):
        raise ValueError("target_weights cannot contain negative weights")
    if sum(target_weights.values()) > 1.0 + 1e-9:
        raise ValueError("target_weights must sum to <= 1.0")
    if not math.isfinite(drift_threshold) or drift_threshold < 0:
        raise ValueError("drift_threshold must be a finite, non-negative number")
    if not math.isfinite(cash):
        raise ValueError("cash must be a finite number")
    if any(pr is not None and not math.isfinite(pr) for pr in prices.values()):
        raise ValueError("prices must be finite numbers")

    # Parse holdings once, validating qty as we go.
    held_qty: dict[str, float] = {}
    for p in positions:
        symbol = p["symbol"]
        qty = float(p["qty"])
        if not math.isfinite(qty):
            raise ValueError(f"non-finite qty for {symbol}")
        held_qty[symbol] = held_qty.get(symbol, 0.0) + qty

    # A held position with no usable price can't be safely priced into the
    # total -- and excluding it would understate the portfolio and mis-size
    # trades for every *other* symbol too. Refuse to guess for anyone rather
    # than partially rebalance against a corrupted total.
    missing_price_symbols = sorted(
        symbol for symbol in held_qty
        if prices.get(symbol) is None or prices.get(symbol, 0) <= 0
    )
    if missing_price_symbols:
        return {
            "total_value": None,
            "trades": [],
            "missing_price_symbols": missing_price_symbols,
            "note": (
                "Cannot compute a reliable rebalance: no usable price for "
                f"{', '.join(missing_price_symbols)}. Excluding it would "
                "understate the portfolio and mis-size every other trade."
            ),
        }

    current_values = {symbol: qty * prices[symbol] for symbol, qty in held_qty.items()}
    total_value = cash + sum(current_values.values())

    if total_value <= 0:
        return {"total_value": 0.0, "trades": [], "note": "No portfolio value to rebalance."}

    all_symbols = sorted(set(target_weights) | set(current_values))
    trades = []

    for symbol in all_symbols:
        price = prices.get(symbol)
        target_weight = target_weights.get(symbol, 0.0)

        if price is None or price <= 0:
            # Only reachable for a target-only symbol we don't currently
            # hold and have no quote for -- can't size a buy, so flag it.
            current_weight = current_values.get(symbol, 0.0) / total_value
            trades.append({
                "symbol": symbol,
                "action": "skipped_no_price",
                "current_weight": round(current_weight, 4),
                "target_weight": round(target_weight, 4),
                "drift": round(current_weight - target_weight, 4),
            })
            continue

        current_value = current_values.get(symbol, 0.0)
        current_weight = current_value / total_value
        drift = current_weight - target_weight

        if abs(drift) <= drift_threshold:
            continue

        target_value = target_weight * total_value
        trade_value = target_value - current_value
        trade_qty = trade_value / price

        qty_final = round(abs(trade_qty), 4)
        if trade_qty < 0:
            # Selling -- never round up past what's actually held.
            qty_final = min(qty_final, held_qty.get(symbol, 0.0))
        if qty_final <= 0:
            # Rounded down to nothing -- not a real order, skip it.
            continue
        est_value = round(qty_final * price, 2)

        trades.append({
            "symbol": symbol,
            "action": "buy" if trade_qty > 0 else "sell",
            "qty": qty_final,
            "est_value": est_value,
            "current_weight": round(current_weight, 4),
            "target_weight": round(target_weight, 4),
            "drift": round(drift, 4),
        })

    # A per-symbol drift filter can retain a buy while suppressing smaller
    # offsetting sells, proposing more spend than cash + sells actually
    # cover. Surface that instead of silently returning an unfunded plan.
    buy_total = sum(t["est_value"] for t in trades if t.get("action") == "buy")
    sell_total = sum(t["est_value"] for t in trades if t.get("action") == "sell")
    shortfall = round(buy_total - sell_total - cash, 2)

    result = {"total_value": round(total_value, 2), "trades": trades}
    if shortfall > 0:
        result["funding_shortfall"] = shortfall
        result["note"] = (
            f"Proposed buys exceed available cash plus proposed sell proceeds "
            f"by ${shortfall:,.2f}. Review allocation or reduce buy sizing before approving."
        )
    return result


if __name__ == "__main__":
    import json

    # Sample data shaped like what Alpaca's get_positions / get_stock_quote
    # tools actually return (qty as a string) -- swap for real tool output.
    sample_positions = [
        {"symbol": "AAPL", "qty": "100"},
        {"symbol": "MSFT", "qty": "50"},
        {"symbol": "SPY", "qty": "20"},
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