"""
Real, computed market data — not web-searched, not model-estimated.

This module fetches actual historical price/volume data and computes
statistics directly, so both agents reason over real numbers rather than
whatever a search result happens to report. If this fails for any reason
(ticker not found, data source unavailable), callers should degrade
gracefully and say so in the report rather than crash the whole run.

Uses yfinance, which works against Yahoo Finance and supports Indian
tickers with .NS (NSE) / .BO (BSE) suffixes as well as US tickers.

Note: this was written and syntax-checked in an isolated sandbox without
outbound access to Yahoo Finance, so the live data fetch has NOT been
end-to-end tested against the real API. Verify it works on your first
real workflow run, and check GitHub Actions logs if it doesn't - Yahoo's
unofficial API occasionally changes shape.
"""

import math

import yfinance as yf


def _benchmark_for(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "^NSEI"  # Nifty 50
    return "^GSPC"  # S&P 500, reasonable default for US and unrecognized tickers


def fetch_and_compute(ticker: str, period: str = "2y") -> dict:
    """Returns a dict of real computed stats, or a dict with 'error' set if
    the fetch/compute failed. Never raises - callers should check for the
    'error' key rather than wrapping this in their own try/except."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 30:
            return {"error": f"Insufficient price history returned for {ticker}"}

        closes = hist["Close"].tolist()
        volumes = hist["Volume"].tolist() if "Volume" in hist else []
        dates = [d.strftime("%Y-%m-%d") for d in hist.index]

        result = {
            "ticker": ticker,
            "data_points": len(closes),
            "date_range": f"{dates[0]} to {dates[-1]}",
            "last_close": round(closes[-1], 2),
        }

        result.update(_moving_averages(closes))
        result.update(_rsi(closes))
        result.update(_macd(closes))
        result.update(_volatility_and_drawdown(closes))
        result.update(_momentum(closes, dates))
        result.update(_volume_stats(volumes))

        beta_result = _beta(ticker, closes, dates, period)
        result.update(beta_result)

        return result
    except Exception as e:  # noqa: BLE001 - deliberately broad, this must never crash the pipeline
        return {"error": f"Market data fetch/compute failed: {e}"}


def _sma(closes: list, window: int):
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def _moving_averages(closes: list) -> dict:
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    out = {"sma_50": round(sma50, 2) if sma50 else None,
           "sma_200": round(sma200, 2) if sma200 else None}
    if sma50 and sma200:
        out["trend_signal"] = (
            "above both SMAs, 50>200 (uptrend)" if closes[-1] > sma50 > sma200 else
            "below both SMAs, 50<200 (downtrend)" if closes[-1] < sma50 < sma200 else
            "mixed / no clean trend"
        )
    else:
        out["trend_signal"] = "insufficient history for 200-day SMA"
    return out


def _rsi(closes: list, window: int = 14) -> dict:
    if len(closes) < window + 1:
        return {"rsi_14": None}
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = deltas[-window:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0:
        return {"rsi_14": 100.0}
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return {"rsi_14": round(rsi, 1)}


def _ema_series(closes: list, span: int) -> list:
    k = 2 / (span + 1)
    ema = [closes[0]]
    for price in closes[1:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema


def _macd(closes: list) -> dict:
    if len(closes) < 35:
        return {"macd_signal_cross": None}
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = _ema_series(macd_line, 9)
    latest_diff = macd_line[-1] - signal_line[-1]
    prev_diff = macd_line[-2] - signal_line[-2]
    if prev_diff <= 0 < latest_diff:
        cross = "bullish cross in the last session"
    elif prev_diff >= 0 > latest_diff:
        cross = "bearish cross in the last session"
    elif latest_diff > 0:
        cross = "MACD above signal (no fresh cross)"
    else:
        cross = "MACD below signal (no fresh cross)"
    return {"macd_signal_cross": cross}


def _volatility_and_drawdown(closes: list) -> dict:
    returns = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes))]
    if not returns:
        return {"annualized_volatility_pct": None, "max_drawdown_pct": None}
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    daily_std = math.sqrt(variance)
    annualized_vol = daily_std * math.sqrt(252) * 100

    peak = closes[0]
    max_dd = 0.0
    for price in closes:
        peak = max(peak, price)
        dd = (price - peak) / peak
        max_dd = min(max_dd, dd)

    return {
        "annualized_volatility_pct": round(annualized_vol, 1),
        "max_drawdown_pct": round(max_dd * 100, 1),
    }


def _momentum(closes: list, dates: list) -> dict:
    def pct_change_over(trading_days):
        if len(closes) <= trading_days:
            return None
        return round((closes[-1] / closes[-trading_days - 1] - 1) * 100, 1)

    return {
        "return_3m_pct": pct_change_over(63),
        "return_6m_pct": pct_change_over(126),
        "return_12m_pct": pct_change_over(252),
    }


def _volume_stats(volumes: list) -> dict:
    if not volumes or len(volumes) < 20:
        return {"volume_vs_20d_avg_pct": None}
    avg20 = sum(volumes[-20:]) / 20
    if avg20 == 0:
        return {"volume_vs_20d_avg_pct": None}
    latest = volumes[-1]
    return {"volume_vs_20d_avg_pct": round((latest / avg20 - 1) * 100, 1)}


def _beta(ticker: str, closes: list, dates: list, period: str) -> dict:
    try:
        bench_ticker = _benchmark_for(ticker)
        bench = yf.Ticker(bench_ticker).history(period=period, auto_adjust=True)
        if bench is None or bench.empty:
            return {"beta_vs_benchmark": None, "beta_benchmark_used": bench_ticker}

        bench_closes_by_date = {d.strftime("%Y-%m-%d"): c for d, c in zip(bench.index, bench["Close"])}
        paired = [(closes[i], dates[i]) for i in range(len(closes)) if dates[i] in bench_closes_by_date]
        if len(paired) < 30:
            return {"beta_vs_benchmark": None, "beta_benchmark_used": bench_ticker}

        stock_series = [p[0] for p in paired]
        bench_series = [bench_closes_by_date[p[1]] for p in paired]

        stock_returns = [(stock_series[i] / stock_series[i - 1]) - 1 for i in range(1, len(stock_series))]
        bench_returns = [(bench_series[i] / bench_series[i - 1]) - 1 for i in range(1, len(bench_series))]

        mean_s = sum(stock_returns) / len(stock_returns)
        mean_b = sum(bench_returns) / len(bench_returns)
        cov = sum((s - mean_s) * (b - mean_b) for s, b in zip(stock_returns, bench_returns)) / len(stock_returns)
        var_b = sum((b - mean_b) ** 2 for b in bench_returns) / len(bench_returns)
        if var_b == 0:
            return {"beta_vs_benchmark": None, "beta_benchmark_used": bench_ticker}

        beta = cov / var_b
        return {"beta_vs_benchmark": round(beta, 2), "beta_benchmark_used": bench_ticker}
    except Exception as e:  # noqa: BLE001
        return {"beta_vs_benchmark": None, "beta_benchmark_used": None, "beta_error": str(e)}


def format_for_prompt(stats: dict) -> str:
    """Render computed stats as a clean block to inject into the model's
    user message, clearly labeled as real computed data, not sourced text."""
    if stats.get("error"):
        return (f"[Computed market data unavailable: {stats['error']}. "
                f"Proceed using web-searched figures only, and say plainly "
                f"in the report that locally-computed technical/quant stats "
                f"could not be obtained for this run.]")

    lines = ["REAL COMPUTED MARKET DATA (calculated directly from price history, "
             "not sourced from a website - treat these as ground truth over any "
             "conflicting web-searched figure):"]
    for key, val in stats.items():
        if key in ("ticker",):
            continue
        lines.append(f"  - {key}: {val}")
    return "\n".join(lines)
