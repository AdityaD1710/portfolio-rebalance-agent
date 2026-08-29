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
    """Round a non-negative quantity DOWN to 4 decimals, never up.

    Used for anything derived from a cap or a funding constraint (sell
    quantities capped at holdings, buy quantities scaled to available
    funds) so that rounding can only shrink the trade, never grow it
    past the limit it was just constrained to.
    """
    if x <= 0:
        return 0.0
    try:
        return math.floor(x * _QTY_SCALE + _FLOOR_EPS) / _QTY_SCALE
    except OverflowError:
        # x is large enough that x * _QTY_SCALE overflows to infinity
        # (only possible near float's ~1.8e308 ceiling). A float at that
        # magnitude has no fractional precision left to floor away in
        # the first place, so return it unchanged -- still satisfies
        # "never increases the quantity".
        return x


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
    cash: uninvested cash, counts toward total portfolio value. Must be
          non-negative -- this model has no concept of margin or debt.
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
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("cash must be a non-negative finite number")
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

    # First pass: raw (unrounded, signed) trade candidates.
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
        trade_value = target_value - current_value  # signed: + buy, - sell
        trade_qty = trade_value / price  # signed

        raw_trades.append({
            "symbol": symbol,
            "price": price,
            "trade_qty": trade_qty,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "drift": drift,
        })

    # Finalize sells FIRST, at the same executable quantities that will
    # actually be returned. Cap at held qty, then floor (never round up)
    # to 4 decimals -- a capped-then-rounded-up sell could exceed the
    # position, and funding must be computed from a quantity that's
    # actually fillable, not one that overstates it.
    sells = [t for t in raw_trades if "trade_qty" in t and t["trade_qty"] < 0]
    buys = [t for t in raw_trades if "trade_qty" in t and t["trade_qty"] > 0]

    for t in sells:
        held = held_qty.get(t["symbol"], 0.0)
        capped = min(abs(t["trade_qty"]), held)
        t["qty_final"] = _floor_qty(capped)
        t["est_value"] = round(t["qty_final"] * t["price"], 2) if t["qty_final"] > 0 else 0.0

    # Funding must use the exact (unrounded-to-cent) proceeds each sell
    # realizes -- est_value is cent-rounded for display and can round UP
    # by up to half a cent per sell, which would let buys draw on proceeds
    # that don't actually exist.
    sell_total = sum(t["qty_final"] * t["price"] for t in sells if t["qty_final"] > 0)
    available_funds = cash + sell_total  # cash >= 0 and sell_total >= 0, so this is always >= 0

    buy_total = sum(t["trade_qty"] * t["price"] for t in buys)

    # A per-symbol drift filter can retain a buy while suppressing smaller
    # offsetting sells, so proposed buys can exceed cash + executable sell
    # proceeds. Scale every buy down proportionally to fit, rather than
    # either overspending or refusing to act. available_funds is guaranteed
    # non-negative (cash is validated non-negative, sell_total is a sum of
    # non-negative values), so scale can never go negative and flip a buy
    # into a sell.
    scale_note = None
    if buy_total > available_funds and buy_total > 0:
        scale = available_funds / buy_total
        scale_note = (
            f"Buys scaled to {scale:.1%} of full target to fit available "
            f"cash (${available_funds:,.2f} available vs ${buy_total:,.2f} needed)."
        )
        for t in buys:
            t["trade_qty"] *= scale

    # Finalize buys: floor (never round up) to 4 decimals. Scaling
    # constrains the *unrounded* total to available_funds exactly; rounding
    # each buy independently to nearest can push the aggregate back over
    # budget. Flooring every buy can only shrink it, so the rounded
    # aggregate is guaranteed <= the unrounded scaled aggregate <=
    # available_funds.
    for t in buys:
        t["qty_final"] = _floor_qty(t["trade_qty"])
        t["est_value"] = round(t["qty_final"] * t["price"], 2) if t["qty_final"] > 0 else 0.0

    # Second pass: assemble output from the finalized quantities.
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


if __name__ == "__main__":
    import json

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