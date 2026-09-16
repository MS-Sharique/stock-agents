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


# Sector/industry keyword -> a real, liquid sector benchmark index/ETF.
# Indian sectoral index tickers (^CNX*, ^NSEBANK) are Yahoo Finance's
# standard symbols for NSE's official sectoral indices, to the best of my
# knowledge - these could NOT be verified against a live network in the
# sandbox this was built in (same limitation as the main price fetch).
# Verify on first real run; if a sector fetch fails, this degrades
# gracefully to the broad index rather than breaking the whole run.
_INDIA_SECTOR_MAP = {
    "information technology": "^CNXIT", "software": "^CNXIT", "it services": "^CNXIT",
    "pharmaceutical": "^CNXPHARMA", "biotechnology": "^CNXPHARMA", "drug": "^CNXPHARMA",
    "auto": "^CNXAUTO", "vehicle": "^CNXAUTO",
    "bank": "^NSEBANK", "financial services": "^CNXFIN",
    "consumer defensive": "^CNXFMCG", "packaged foods": "^CNXFMCG", "household": "^CNXFMCG",
    "metal": "^CNXMETAL", "steel": "^CNXMETAL", "mining": "^CNXMETAL",
    "real estate": "^CNXREALTY", "realty": "^CNXREALTY",
    "energy": "^CNXENERGY", "oil": "^CNXENERGY", "gas": "^CNXENERGY",
    "utilities": "^CNXENERGY", "power": "^CNXENERGY",
    "infrastructure": "^CNXINFRA", "construction": "^CNXINFRA",
    "media": "^CNXMEDIA", "entertainment": "^CNXMEDIA",
}

# US sector -> SPDR Select Sector ETF. These are well-established, highly
# liquid, and reliably present in Yahoo Finance - much higher confidence
# than the Indian sectoral index mapping above.
_US_SECTOR_MAP = {
    "technology": "XLK", "software": "XLK", "semiconductor": "XLK",
    "financial services": "XLF", "bank": "XLF", "insurance": "XLF",
    "energy": "XLE", "oil": "XLE", "gas": "XLE",
    "healthcare": "XLV", "pharmaceutical": "XLV", "biotechnology": "XLV",
    "consumer cyclical": "XLY", "consumer discretionary": "XLY", "retail": "XLY", "auto": "XLY",
    "consumer defensive": "XLP", "consumer staples": "XLP", "packaged foods": "XLP",
    "industrials": "XLI", "aerospace": "XLI", "manufacturing": "XLI",
    "basic materials": "XLB", "chemicals": "XLB", "metal": "XLB", "mining": "XLB",
    "utilities": "XLU",
    "real estate": "XLRE",
    "communication services": "XLC", "telecom": "XLC", "media": "XLC",
}


def _sector_benchmark_for(ticker: str, sector: str, industry: str):
    """Match this stock's sector/industry to a real sector benchmark
    index/ETF. Returns None if no confident match is found, rather than
    guessing - callers should fall back to broad-index-only in that case."""
    t = ticker.upper()
    is_indian = t.endswith(".NS") or t.endswith(".BO")
    mapping = _INDIA_SECTOR_MAP if is_indian else _US_SECTOR_MAP
    haystack = f"{(sector or '').lower()} {(industry or '').lower()}"
    for keyword, bench_ticker in mapping.items():
        if keyword in haystack:
            return bench_ticker
    return None


def fetch_and_compute(ticker: str, period: str = "2y") -> dict:
    """Returns a dict of real computed stats, or a dict with 'error' set if
    the fetch/compute failed. Never raises - callers should check for the
    'error' key rather than wrapping this in their own try/except.

    Also stashes the raw OHLCV under '_raw_ohlcv' (leading underscore =
    not for direct prompt inclusion, see format_for_prompt) so the
    backtest module can reuse this same fetch instead of hitting Yahoo
    Finance a second time."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 30:
            return {"error": f"Insufficient price history returned for {ticker}"}

        closes = hist["Close"].tolist()
        highs = hist["High"].tolist() if "High" in hist else closes
        lows = hist["Low"].tolist() if "Low" in hist else closes
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
        result.update(_atr(hist))
        result.update(_adx(highs, lows, closes))
        result.update(_sharpe(closes))
        result["_raw_ohlcv"] = {"dates": dates, "closes": closes, "highs": highs,
                                 "lows": lows, "volumes": volumes}

        broad_bench = _benchmark_for(ticker)
        result.update(_beta_and_relative_strength(closes, dates, broad_bench, period, "broad"))

        sector_info = _fetch_sector_info(stock)
        result["sector"] = sector_info.get("sector")
        result["industry"] = sector_info.get("industry")
        sector_bench = _sector_benchmark_for(ticker, sector_info.get("sector"), sector_info.get("industry"))
        if sector_bench and sector_bench != broad_bench:
            result.update(_beta_and_relative_strength(closes, dates, sector_bench, period, "sector"))
        else:
            result["sector_benchmark_used"] = None
            result["sector_note"] = ("No confident sector benchmark match found - only broad-index "
                                      "beta/relative-strength is available for this stock.")

        result.update(_fetch_fundamentals(stock))

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


def _atr(hist, window: int = 14) -> dict:
    """Average True Range - a real, computed volatility-per-day figure used
    to derive a stop-loss distance grounded in the stock's own actual
    recent price behavior, not a guessed percentage."""
    try:
        highs = hist["High"].tolist()
        lows = hist["Low"].tolist()
        closes = hist["Close"].tolist()
        if len(closes) < window + 1:
            return {"atr_14": None}
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)
        atr = sum(true_ranges[-window:]) / window
        return {"atr_14": round(atr, 2), "atr_14_pct_of_price": round(atr / closes[-1] * 100, 1)}
    except Exception:  # noqa: BLE001
        return {"atr_14": None}


def _adx(highs: list, lows: list, closes: list, window: int = 14) -> dict:
    """Average Directional Index - measures TREND STRENGTH, distinct from
    direction. A stock can be "in an uptrend" by the SMA50>SMA200 test
    while that trend is actually weak/choppy - ADX catches that. Below
    ~20 is conventionally read as a weak/absent trend regardless of
    direction; above ~25 as a trend with real conviction behind it."""
    try:
        n = len(closes)
        if n < window * 2 + 1:
            return {"adx_14": None}

        plus_dm, minus_dm, tr_list = [], [], []
        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
            minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
            tr_list.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))

        def wilder_smooth(series, window):
            smoothed = [sum(series[:window])]
            for v in series[window:]:
                smoothed.append(smoothed[-1] - (smoothed[-1] / window) + v)
            return smoothed

        tr_smooth = wilder_smooth(tr_list, window)
        plus_dm_smooth = wilder_smooth(plus_dm, window)
        minus_dm_smooth = wilder_smooth(minus_dm, window)

        dx_values = []
        for tr_s, pdm_s, mdm_s in zip(tr_smooth, plus_dm_smooth, minus_dm_smooth):
            if tr_s == 0:
                continue
            plus_di = 100 * pdm_s / tr_s
            minus_di = 100 * mdm_s / tr_s
            di_sum = plus_di + minus_di
            if di_sum == 0:
                continue
            dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

        if len(dx_values) < window:
            return {"adx_14": None}
        adx = sum(dx_values[-window:]) / window
        strength = "strong trend" if adx >= 25 else ("weak/no clear trend" if adx < 20 else "moderate trend")
        return {"adx_14": round(adx, 1), "adx_14_reading": strength}
    except Exception:  # noqa: BLE001
        return {"adx_14": None}


def _sharpe(closes: list, risk_free_annual_pct: float = 6.5) -> dict:
    """Risk-adjusted return: excess return per unit of volatility taken to
    get it. A stock with a high raw return but very high volatility can
    have a WORSE Sharpe than a steadier, lower-return one - this is the
    standard way to compare "was the return worth the ride" rather than
    looking at return in isolation."""
    try:
        returns = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes))]
        if len(returns) < 30:
            return {"sharpe_ratio_annualized": None}
        mean_daily = sum(returns) / len(returns)
        variance = sum((r - mean_daily) ** 2 for r in returns) / len(returns)
        daily_std = variance ** 0.5
        if daily_std == 0:
            return {"sharpe_ratio_annualized": None}
        annualized_return = mean_daily * 252
        annualized_vol = daily_std * (252 ** 0.5)
        sharpe = (annualized_return - risk_free_annual_pct / 100) / annualized_vol
        return {"sharpe_ratio_annualized": round(sharpe, 2)}
    except Exception:  # noqa: BLE001
        return {"sharpe_ratio_annualized": None}


def _beta_and_relative_strength(closes: list, dates: list, bench_ticker: str, period: str,
                                 label: str) -> dict:
    """Computes beta AND relative strength (this stock's return minus the
    benchmark's return, over 3/6/12mo) against a given benchmark ticker.
    `label` prefixes the output keys ("broad" or "sector") so both can
    coexist in the same stats dict without colliding."""
    try:
        bench = yf.Ticker(bench_ticker).history(period=period, auto_adjust=True)
        if bench is None or bench.empty:
            return {f"{label}_beta": None, f"{label}_benchmark_used": bench_ticker}

        bench_closes_by_date = {d.strftime("%Y-%m-%d"): c for d, c in zip(bench.index, bench["Close"])}
        paired = [(closes[i], dates[i]) for i in range(len(closes)) if dates[i] in bench_closes_by_date]
        if len(paired) < 30:
            return {f"{label}_beta": None, f"{label}_benchmark_used": bench_ticker}

        stock_series = [p[0] for p in paired]
        bench_series = [bench_closes_by_date[p[1]] for p in paired]

        stock_returns = [(stock_series[i] / stock_series[i - 1]) - 1 for i in range(1, len(stock_series))]
        bench_returns = [(bench_series[i] / bench_series[i - 1]) - 1 for i in range(1, len(bench_series))]

        mean_s = sum(stock_returns) / len(stock_returns)
        mean_b = sum(bench_returns) / len(bench_returns)
        cov = sum((s - mean_s) * (b - mean_b) for s, b in zip(stock_returns, bench_returns)) / len(stock_returns)
        var_b = sum((b - mean_b) ** 2 for b in bench_returns) / len(bench_returns)

        result = {f"{label}_benchmark_used": bench_ticker}
        result[f"{label}_beta"] = round(cov / var_b, 2) if var_b != 0 else None

        # Relative strength: this stock's return minus the benchmark's
        # return over the same window - positive means outperforming that
        # benchmark specifically, distinct from just "the stock went up".
        for months, trading_days in [(3, 63), (6, 126), (12, 252)]:
            if len(stock_series) > trading_days and len(bench_series) > trading_days:
                stock_ret = (stock_series[-1] / stock_series[-trading_days - 1] - 1) * 100
                bench_ret = (bench_series[-1] / bench_series[-trading_days - 1] - 1) * 100
                result[f"{label}_relative_strength_{months}m_pct"] = round(stock_ret - bench_ret, 1)
        return result
    except Exception as e:  # noqa: BLE001
        return {f"{label}_beta": None, f"{label}_benchmark_used": None, f"{label}_error": str(e)}


def _fetch_sector_info(stock) -> dict:
    """Best-effort sector/industry lookup via yfinance's .info - this field
    is generally reliable even when deeper fundamentals aren't."""
    try:
        info = stock.info or {}
        return {"sector": info.get("sector"), "industry": info.get("industry")}
    except Exception:  # noqa: BLE001
        return {"sector": None, "industry": None}


def _fetch_fundamentals(stock) -> dict:
    """Best-effort fundamental data via yfinance's .info and quarterly
    financials. Unlike price/technical data, this is NOT treated as
    unconditional ground truth in the prompt - coverage for Indian tickers
    especially can be incomplete or lag behind the actual latest quarter.
    Explicitly flags staleness so the model knows to prefer a fresher
    web-search figure when the computed one looks out of date."""
    out = {}
    try:
        info = stock.info or {}
        candidates = {
            "fund_market_cap": info.get("marketCap"),
            "fund_trailing_pe": info.get("trailingPE"),
            "fund_total_revenue_ttm": info.get("totalRevenue"),
            "fund_total_debt": info.get("totalDebt"),
            "fund_total_cash": info.get("totalCash"),
            "fund_return_on_equity": info.get("returnOnEquity"),
        }
        out.update({k: v for k, v in candidates.items() if v is not None})
    except Exception as e:  # noqa: BLE001
        out["fund_error"] = f"'.info' lookup failed: {e}"

    as_of_date = None
    try:
        qf = stock.quarterly_financials
        if qf is not None and not qf.empty:
            as_of = qf.columns[0]
            as_of_date = as_of.date() if hasattr(as_of, "date") else None
    except Exception:  # noqa: BLE001
        pass

    if as_of_date:
        import datetime as _dt
        days_old = (_dt.date.today() - as_of_date).days
        out["fund_as_of_quarter"] = str(as_of_date)
        out["fund_data_age_days"] = days_old
        # Indian companies typically report within ~45 days of quarter-end;
        # well beyond one full quarter-cycle (120 days) is a real staleness flag.
        out["fund_data_possibly_stale"] = days_old > 120
    else:
        out["fund_as_of_quarter"] = None
        out["fund_data_possibly_stale"] = True
        out["fund_note"] = "No quarterly financials date available from this data source - recency unknown, treat as possibly stale."

    return out




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
             "conflicting web-searched figure):",
             "NOTE ON RECENCY: 'last_close' is the most recently COMPLETED trading "
             "session's close, from the daily price history. If this analysis is "
             "run while a market is currently open, this will be YESTERDAY's close, "
             "not a live intraday price - say so explicitly if it's relevant (e.g. "
             "if there's been recent major news that might have moved the stock "
             "intraday today, note that this figure may not reflect that yet)."]
    for key, val in stats.items():
        if key.startswith("_") or key == "ticker":
            continue
        lines.append(f"  - {key}: {val}")
    return "\n".join(lines)
