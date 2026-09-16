"""
Real, deterministic backtests against a stock's own historical price data.

Two different things, deliberately not conflated:

1. Short-term signal backtest: mechanically re-applies the SAME trend +
   momentum + volume rule used for the current signal check, at every
   historical point where it's computable, and simulates forward with the
   same ATR-based stop/target logic. This genuinely tests "has this
   mechanical setup worked on this stock before" - a real backtest.

2. Long-term holding-rule backtest: mechanically re-applies the SAME
   "hold >= N months, sell at first point return >= target%" exit rule at
   sampled points across history. This tests the EXIT MECHANICS against
   real price data - it does NOT and CANNOT replay the fundamental/
   governance judgment that would have picked the entry point in the
   first place, since that requires re-running research as it existed at
   a past date, which isn't something this can rigorously do. Report
   output must say this explicitly, every time - this is the single most
   important honesty boundary in this whole module.

Everything here is pure arithmetic on data already fetched by
market_data.py - no additional network calls, no LLM involvement, fully
deterministic and reproducible given the same price history.
"""

import html


def _sma_series(closes: list, window: int) -> list:
    out = [None] * len(closes)
    for i in range(window - 1, len(closes)):
        out[i] = sum(closes[i - window + 1:i + 1]) / window
    return out


def _rsi_series(closes: list, window: int = 14) -> list:
    out = [None] * len(closes)
    if len(closes) < window + 1:
        return out
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    for i in range(window, len(closes)):
        recent = deltas[i - window:i]
        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]
        avg_gain = sum(gains) / window
        avg_loss = sum(losses) / window
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - (100 / (1 + rs))
    return out


def _atr_series(highs: list, lows: list, closes: list, window: int = 14) -> list:
    out = [None] * len(closes)
    true_ranges = [None]
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        true_ranges.append(tr)
    for i in range(window, len(closes)):
        window_trs = [tr for tr in true_ranges[i - window + 1:i + 1] if tr is not None]
        if len(window_trs) == window:
            out[i] = sum(window_trs) / window
    return out


def _volume_avg_series(volumes: list, window: int = 20) -> list:
    out = [None] * len(volumes)
    for i in range(window - 1, len(volumes)):
        out[i] = sum(volumes[i - window + 1:i + 1]) / window
    return out


def backtest_short_term(raw_ohlcv: dict, target_pct: float, max_hold_months: int) -> dict:
    """Walk the full history; at every day where trend+momentum+volume all
    align (same rule as the live signal check), simulate forward with an
    ATR(14)-based stop and the stated target, and record which was hit
    first. Returns aggregate stats, or {'signals_found': 0} if none."""
    closes = raw_ohlcv["closes"]
    highs = raw_ohlcv["highs"]
    lows = raw_ohlcv["lows"]
    volumes = raw_ohlcv["volumes"]
    dates = raw_ohlcv["dates"]
    n = len(closes)

    sma50 = _sma_series(closes, 50)
    sma200 = _sma_series(closes, 200)
    rsi = _rsi_series(closes, 14)
    atr = _atr_series(highs, lows, closes, 14)
    vol_avg20 = _volume_avg_series(volumes, 20)

    max_hold_days = max_hold_months * 21  # ~21 trading days/month

    outcomes = []  # each: {"result": "target"|"stop"|"neither", "return_pct": x, "days": n}

    for i in range(200, n - 1):  # need 200 days of lookback, and at least 1 day forward
        if sma50[i] is None or sma200[i] is None or rsi[i] is None or rsi[i - 1] is None or atr[i] is None or vol_avg20[i] is None:
            continue

        trend_ok = closes[i] > sma50[i] > sma200[i]
        momentum_ok = rsi[i - 1] < 45 <= rsi[i]  # RSI just crossed up through 45
        volume_ok = vol_avg20[i] > 0 and volumes[i] > 1.5 * vol_avg20[i]

        if not (trend_ok and momentum_ok and volume_ok):
            continue

        entry = closes[i]
        stop_price = entry - 1.5 * atr[i]
        target_price = entry * (1 + target_pct / 100)

        horizon_end = min(i + max_hold_days, n - 1)
        result = "neither"
        exit_price = closes[horizon_end]
        days_taken = horizon_end - i

        for j in range(i + 1, horizon_end + 1):
            hit_stop = lows[j] <= stop_price
            hit_target = highs[j] >= target_price
            if hit_stop and hit_target:
                # Both in the same day's range - can't know which came first
                # from daily bars; conservatively assume the stop was hit
                # (the more cautious assumption for reporting accuracy).
                result = "stop"
                exit_price = stop_price
                days_taken = j - i
                break
            elif hit_stop:
                result = "stop"
                exit_price = stop_price
                days_taken = j - i
                break
            elif hit_target:
                result = "target"
                exit_price = target_price
                days_taken = j - i
                break

        return_pct = (exit_price - entry) / entry * 100
        outcomes.append({"result": result, "return_pct": return_pct, "days": days_taken,
                          "date": dates[i]})

    if not outcomes:
        return {"signals_found": 0}

    n_out = len(outcomes)
    target_hits = sum(1 for o in outcomes if o["result"] == "target")
    stop_hits = sum(1 for o in outcomes if o["result"] == "stop")
    neither = n_out - target_hits - stop_hits
    avg_return = sum(o["return_pct"] for o in outcomes) / n_out
    avg_days = sum(o["days"] for o in outcomes) / n_out

    return {
        "signals_found": n_out,
        "target_hit_pct": round(target_hits / n_out * 100, 1),
        "stop_hit_pct": round(stop_hits / n_out * 100, 1),
        "neither_pct": round(neither / n_out * 100, 1),
        "avg_return_pct": round(avg_return, 1),
        "avg_days_to_outcome": round(avg_days, 1),
        "most_recent_signal_date": outcomes[-1]["date"],
    }


def backtest_long_term_holding(raw_ohlcv: dict, target_pct: float, min_hold_months: int,
                                sample_every_days: int = 60) -> dict:
    """At regularly sampled points across the available history, simulate:
    hold for at least min_hold_months, then sell at the first point total
    return reaches target_pct (checked from min_hold_months onward), or
    exit at the end of available data if never reached. This tests the
    EXIT RULE mechanics only - see module docstring for why this cannot
    and does not replay the fundamental entry judgment."""
    closes = raw_ohlcv["closes"]
    dates = raw_ohlcv["dates"]
    n = len(closes)
    min_hold_days = min_hold_months * 21

    outcomes = []

    for i in range(0, n - min_hold_days, sample_every_days):
        entry = closes[i]
        target_price = entry * (1 + target_pct / 100)
        result = "neither"
        exit_price = closes[-1]
        days_taken = n - 1 - i

        for j in range(i + min_hold_days, n):
            if closes[j] >= target_price:
                result = "target"
                exit_price = closes[j]
                days_taken = j - i
                break

        if result == "neither":
            # Never hit target even by the end of available data - record
            # the actual return achieved by the last available price.
            pass

        return_pct = (exit_price - entry) / entry * 100
        outcomes.append({"result": result, "return_pct": return_pct, "days": days_taken,
                          "date": dates[i]})

    if not outcomes:
        return {"windows_tested": 0}

    n_out = len(outcomes)
    target_hits = sum(1 for o in outcomes if o["result"] == "target")
    avg_return = sum(o["return_pct"] for o in outcomes) / n_out
    avg_days_to_target = (
        sum(o["days"] for o in outcomes if o["result"] == "target") / target_hits
        if target_hits else None
    )

    return {
        "windows_tested": n_out,
        "target_hit_pct": round(target_hits / n_out * 100, 1),
        "avg_return_pct_all_windows": round(avg_return, 1),
        "avg_months_to_target_when_hit": round(avg_days_to_target / 21, 1) if avg_days_to_target else None,
    }


MIN_SIGNALS_FOR_STATS = 5  # below this, a "hit rate %" is misleadingly
                            # precise-looking for what's actually a tiny
                            # sample - report the raw count instead


def format_short_term_backtest_html(stats: dict, target_pct: float) -> str:
    if stats.get("signals_found", 0) == 0:
        return (
            "<h3>Historical Track Record for This Signal</h3>"
            "<p>This exact trend+momentum+volume signal did not occur "
            "anywhere else in the available price history for this stock "
            "(or there wasn't enough history to check). No historical "
            "hit-rate is available - this may be the first time this "
            "specific setup has appeared, or the stock is too newly "
            "listed to have enough data.</p>"
        )
    if stats["signals_found"] < MIN_SIGNALS_FOR_STATS:
        return f"""
<h3>Historical Track Record for This Signal</h3>
<p>This signal occurred only <strong>{stats['signals_found']} time(s)</strong>
in the available price history (most recently around
{stats['most_recent_signal_date']}). <strong>That's too small a sample to
report a meaningful hit-rate percentage</strong> - a "67% hit rate" from
3 instances is not statistically informative, even though the arithmetic
is correct. What did happen: an average realized return of
{stats['avg_return_pct']:+.1f}% over an average of {stats['avg_days_to_outcome']:.0f}
trading days - but treat this as a small number of individual data points,
not a rate.</p>
"""
    return f"""
<h3>Historical Track Record for This Signal (real backtest, not a model estimate)</h3>
<p>This exact trend+momentum+volume rule was mechanically re-applied to
this stock's own price history. It triggered <strong>{stats['signals_found']}
time(s)</strong> before now (most recently around {stats['most_recent_signal_date']}) -
enough instances to report a rate, though still specific to this one
stock's own history, not a broad statistical sample. Of those instances:</p>
<table class="report-table">
<tr><th>Outcome</th><th>% of past signals</th></tr>
<tr><td>Reached the {target_pct:.0f}% target before the stop-loss</td><td>{stats['target_hit_pct']}%</td></tr>
<tr><td>Hit the stop-loss first</td><td>{stats['stop_hit_pct']}%</td></tr>
<tr><td>Neither within the hold window</td><td>{stats['neither_pct']}%</td></tr>
</table>
<p>Average realized return across all past instances: <strong>{stats['avg_return_pct']:+.1f}%</strong>,
average {stats['avg_days_to_outcome']:.0f} trading days to an outcome.
<strong>This is a real, computed historical hit-rate for this specific
mechanical rule on this specific stock - it is not a guarantee.</strong>
Three things this does NOT account for: (1) brokerage, STT, and slippage,
which would reduce every realized return shown here somewhat; (2) a market
regime shift - a rule that worked over this stock's available price
history may not hold if broader market conditions change; (3) daily price
bars can't always tell whether the stop or target was hit first if both
happened the same day, which was conservatively counted as the stop being
hit.</p>
"""


def format_long_term_backtest_html(stats: dict, target_pct: float, min_hold_months: int) -> str:
    if stats.get("windows_tested", 0) == 0:
        return (
            "<h3>Historical Track Record for This Holding Rule</h3>"
            "<p>Not enough price history was available to test this "
            "holding rule on this stock.</p>"
        )
    if stats["windows_tested"] < MIN_SIGNALS_FOR_STATS:
        return f"""
<h3>Historical Track Record for This Holding Rule</h3>
<p><strong>Only {stats['windows_tested']} historical entry window(s) could be
tested</strong> - too few to report a meaningful hit-rate percentage.
Average return across the window(s) tested was
{stats['avg_return_pct_all_windows']:+.1f}%, but treat this as a small
number of individual observations, not a rate. Same scope note as always:
this tests the exit-rule mechanics only, not any fundamental judgment.</p>
"""
    avg_months = stats.get("avg_months_to_target_when_hit")
    months_line = f"an average of {avg_months:.1f} months to reach it when it did" if avg_months else "it never reached the target within available history in the windows tested"
    return f"""
<h3>Historical Track Record for This Holding Rule (real backtest, not a model estimate)</h3>
<p><strong>Important scope note: this tests the mechanical exit rule only
(hold {min_hold_months}+ months, sell at {target_pct:.0f}%+) against this
stock's own real price history - it does NOT and cannot replay the
fundamental/governance judgment used to justify entering in the first
place, since that would require re-running research as it existed at each
past date. Treat this as "how has this stock's own price behaved under
this holding rule," not "how often was our analysis right."</strong></p>
<p>Sampling entry points across the available price history and applying
the rule: out of <strong>{stats['windows_tested']}</strong> historical
entry windows tested, <strong>{stats['target_hit_pct']}%</strong> reached
the {target_pct:.0f}% target at some point after the minimum hold period,
with {months_line}. Average return across ALL windows tested (including
ones that never reached target) was
<strong>{stats['avg_return_pct_all_windows']:+.1f}%</strong>.
This reflects this stock's own historical volatility and growth pattern
under this specific rule over the available data window - it does not
account for brokerage/STT/capital-gains-tax drag, and a market regime
shift means this stock's next {min_hold_months}+ months are not
guaranteed to resemble its past ones. Not a forecast, and not evidence
about the quality of any particular entry judgment.</p>
"""
