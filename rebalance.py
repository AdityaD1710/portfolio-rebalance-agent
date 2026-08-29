"""
Target-allocation-drift rebalance calculator.

Given current positions, live prices, and target weights, this computes
which trades would bring the portfolio back to target -- flagging only
positions that have drifted beyond a threshold, rather than rebalancing
everything to the decimal every time (which would be noisy and unrealistic).

If proposed buys would exceed available cash plus proposed sell proceeds,
every buy is scaled down proportionally so the resulting trade plan is
always fundable -- rather than either overspending or refusing to act.
Funding is computed from the same rounded, executable sell quantities
that are actually returned, not from raw pre-rounding sell values, and
all quantities are rounded toward zero (never up) so that neither a
capped sell nor a scaled buy can end up larger than what it was capped
or scaled to.

This is deliberately simple on purpose: no Sharpe ratio, no mean-variance
optimization. Get this loop (real data -> compute -> proposed trade ->
approval) working end to end first. Fancier math is a stretch goal, not
a prerequisite.
"""

from __future__ import annotations

import math

_QTY_DECIMALS = 4
_QTY_SCALE = 10 ** _QTY_DECIMALS
_FLOOR_EPS = 1e-9  # guards against float imprecision underflooring a value
                     # that is mathematically exact at the 4-decimal boundary


def _floor_qty(x: float) -> float:
    """Round a non-negative quantity DOWN to 4 decimals, never up."""
    if x <= 0:
        return 0.0
    try:
        return math.floor(x * _QTY_SCALE + _FLOOR_EPS) / _QTY_SCALE
    except OverflowError:
        return x


def compute_rebalance(
    positions: list[dict],
    prices: dict[str, float],
    target_weights: dict[str, float],
    cash: float = 0.0,
    drift_threshold: float = 0.05,
) -> dict:
    if any(not math.isfinite(w) for w in target_weights.values()):
        raise ValueError("target_weights must be finite numbers")
    if any(w < 0 for w in target_weights.values()):
        raise ValueError("target_weights cannot contain negative weights")
    if sum(target_weights.values()) > 1.0 + 1e-9:
        raise ValueError("target_weights must sum to <= 1.0")
    if not math.isfinite(drift_threshold) or drift_threshold < 0:
        raise ValueError("drift_threshold must be a finite, non-negative number")
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("cash must be a non-negative finite number")

    normalized_prices: dict[str, float | None] = {}
    for symbol, pr in prices.items():
        if pr is None:
            normalized_prices[symbol] = None
            continue
        try:
            pr_f = float(pr)
        except (TypeError, ValueError):
            raise ValueError(f"price for {symbol} is not a number: {pr!r}")
        if not math.isfinite(pr_f):
            raise ValueError(f"non-finite price for {symbol}")
        normalized_prices[symbol] = pr_f
    prices = normalized_prices

    held_qty: dict[str, float] = {}
    for i, p in enumerate(positions):
        if "symbol" not in p:
            raise ValueError(f"positions[{i}] is missing required key 'symbol'")
        symbol = p["symbol"]
        if "qty" not in p:
            raise ValueError(f"positions[{i}] ({symbol}) is missing required key 'qty'")
        try:
            qty = float(p["qty"])
        except (TypeError, ValueError):
            raise ValueError(f"qty for {symbol} is not a number: {p['qty']!r}")
        if not math.isfinite(qty):
            raise ValueError(f"non-finite qty for {symbol}")
        if qty < 0:
            raise ValueError(f"negative qty for {symbol}: {qty} (short positions are not supported)")
        held_qty[symbol] = held_qty.get(symbol, 0.0) + qty

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

    raw_trades = []
    for symbol in all_symbols:
        price = prices.get(symbol)
        target_weight = target_weights.get(symbol, 0.0)

        if price is None or price <= 0:
            current_weight = current_values.get(symbol, 0.0) / total_value
            raw_trades.append({
                "symbol": symbol,
                "skipped_no_price": True,
                "current_weight": current_weight,
                "target_weight": target_weight,
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

        raw_trades.append({
            "symbol": symbol,
            "price": price,
            "trade_qty": trade_qty,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "drift": drift,
        })

    sells = [t for t in raw_trades if "trade_qty" in t and t["trade_qty"] < 0]
    buys = [t for t in raw_trades if "trade_qty" in t and t["trade_qty"] > 0]

    for t in sells:
        held = held_qty.get(t["symbol"], 0.0)
        capped = min(abs(t["trade_qty"]), held)
        t["qty_final"] = _floor_qty(capped)
        t["est_value"] = round(t["qty_final"] * t["price"], 2) if t["qty_final"] > 0 else 0.0

    sell_total = sum(t["qty_final"] * t["price"] for t in sells if t["qty_final"] > 0)
    available_funds = cash + sell_total

    buy_total = sum(t["trade_qty"] * t["price"] for t in buys)

    scale_note = None
    if buy_total > available_funds and buy_total > 0:
        scale = available_funds / buy_total
        scale_note = (
            f"Buys scaled to {scale:.1%} of full target to fit available "
            f"cash (${available_funds:,.2f} available vs ${buy_total:,.2f} needed)."
        )
        for t in buys:
            t["trade_qty"] *= scale

    for t in buys:
        t["qty_final"] = _floor_qty(t["trade_qty"])
        t["est_value"] = round(t["qty_final"] * t["price"], 2) if t["qty_final"] > 0 else 0.0

    trades = []
    for t in raw_trades:
        if t.get("skipped_no_price"):
            drift = t["current_weight"] - t["target_weight"]
            trades.append({
                "symbol": t["symbol"],
                "action": "skipped_no_price",
                "current_weight": round(t["current_weight"], 4),
                "target_weight": round(t["target_weight"], 4),
                "drift": round(drift, 4),
            })
            continue

        if t["qty_final"] <= 0:
            continue

        trades.append({
            "symbol": t["symbol"],
            "action": "buy" if t["trade_qty"] > 0 else "sell",
            "qty": t["qty_final"],
            "est_value": t["est_value"],
            "current_weight": round(t["current_weight"], 4),
            "target_weight": round(t["target_weight"], 4),
            "drift": round(t["drift"], 4),
        })

    result = {"total_value": round(total_value, 2), "trades": trades}
    if scale_note:
        result["note"] = scale_note
    return result