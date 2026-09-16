#!/usr/bin/env python3
"""
Long-term stock analysis agent.

Exit logic: hold at least MIN_HOLD_MONTHS, sell the first time unrealized
return reaches TARGET_RETURN_PCT, whenever that happens.

Combines three kinds of input, deliberately kept separate:
  1. Real computed market statistics (market_data.py) - deterministic,
     locally computed from actual price history, not model-estimated.
  2. Web search - for qualitative, current, sourced context (financials,
     governance, news) that can't be computed from price data alone.
  3. A historical pattern library baked into the prompt, PLUS an explicit
     instruction to search for genuinely comparable cases in the same
     sector, rather than relying only on the fixed examples.

Usage:
  python analyze_long_term.py "RELIANCE.NS"
"""

import datetime
import os
import sys

import anthropic

import market_data
import report_utils
import backtest

TARGET_RETURN_PCT = float(os.environ.get("TARGET_RETURN_PCT", "50"))
MIN_HOLD_MONTHS = int(os.environ.get("MIN_HOLD_MONTHS", "12"))
INVESTMENT_AMOUNT = os.environ.get("INVESTMENT_AMOUNT", "5000")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = 6000
MAX_CONTINUATIONS = 2  # was 6 - each continuation resends the full growing
                        # conversation, which is what actually drove cost
                        # (confirmed from real usage data: 314K input tokens
                        # across 2 runs, only 14 total searches - the cost
                        # was resent context, not search volume)
SEARCH_MAX_USES = 15   # restored - your usage data showed search was only
                        # $0.14 of $0.98, so 8 saved little cost while
                        # risking cut-off research on complex stocks;
                        # the real cost fix is caching + fewer continuations
HORIZON_SUBDIR = "long"


SYSTEM_PROMPT = f"""You are a seasoned, skeptical equity research analyst.
You are producing an automated research note on a single stock for an
investor whose actual strategy is: HOLD FOR AT LEAST {MIN_HOLD_MONTHS}
MONTHS (a floor, not a target exit date), THEN SELL THE FIRST TIME THE
UNREALIZED RETURN REACHES {TARGET_RETURN_PCT}% OR MORE, whenever that
happens to occur - sooner is preferred but not assumed. The investment
must also be Shariah-compliant.

This means your job is NOT to price a fixed-date target. It is to assess
how PLAUSIBLE a {TARGET_RETURN_PCT}%+ move is at all for this specific
stock, and roughly how long that kind of move has typically taken for
genuinely comparable situations versus stocks that only superficially
resembled them.

You will be given, in the user message, a block of REAL COMPUTED MARKET
DATA (moving averages, RSI, volatility, drawdown, momentum, beta and
relative strength against BOTH a broad market index and this stock's own
sector benchmark where one could be matched, ADX for trend strength, and
an annualized Sharpe ratio) that was calculated directly from actual
price history in Python - not sourced from a website, not estimated by
you. Treat the price/technical fields as ground truth.

You will ALSO be given best-effort FUNDAMENTAL data (market cap, P/E,
revenue, debt, cash, ROE) pulled from a financial data feed - but treat
this differently from the price data: it is explicitly flagged when it
looks stale (fund_data_possibly_stale), and coverage for Indian tickers
especially can be incomplete. Use it as a cross-check and starting point,
not unconditional ground truth - if your own web search finds a more
recent quarter's figures, prefer the fresher one and say explicitly which
source you used and its as-of date. Never silently pick one without
saying so if they conflict. **Report the "last_close" figure from the
computed data block as your current_price by default - not a web-searched
price, which can be stale in unpredictable ways.** Note explicitly, though,
that last_close reflects the most recently COMPLETED session (yesterday's
close if markets are open right now, not a live intraday price) - if
there's major same-day news that could have moved the stock intraday,
say so rather than presenting a known-prior-day figure as if it were live.
Use your web_search tool for everything else: financials, governance,
news, and - importantly - genuinely comparable sector precedents (see
below). Search efficiently: start with a handful of targeted searches
(3-5) covering the highest-value gaps (recent financials, governance
check, one peer comparison, one sector precedent); only search further if
those leave a real gap, rather than searching exhaustively by default.
If different sources give conflicting figures for the same metric, say so
explicitly rather than silently picking one.

## Historical pattern library - use this to ground your judgment, not replace it

Real past cases of Indian small/mid-cap stocks that genuinely delivered
50%+ (often far more) moves shared traits BEFORE the move, not just after:
- Multi-year (not single-quarter) revenue and profit growth, commonly in
  the 15-40%+ range sustained over several years, not one strong quarter
- ROE/ROCE consistently above roughly 15-20%
- A SPECIFIC, NAMEABLE catalyst that the re-rating followed rather than
  preceded - examples from real cases: a debt restructuring/deleveraging
  turnaround (Suzlon Energy), a policy-driven capacity buildout (Dixon
  Technologies riding PLI-scheme electronics manufacturing), a franchise
  or retail-format expansion (Trent's Zudio format, Varun Beverages'
  territory expansion), or a sector-specific order/policy pipeline
  (defense-linked names during the indigenization push, e.g. Taneja
  Aerospace) combined with rising institutional (FII/DII) interest and
  reasonable liquidity/float
- The re-rating (multiple expansion) arrived ALONGSIDE improving
  fundamentals, not as a substitute for them

Real past cases of stocks that LOOKED similar but were look-alike traps,
not genuine setups:
- Price surging on thin float and minimal-to-zero institutional ownership,
  often on a single speculative narrative (litigation outcome, takeover
  rumor) rather than a structural business driver - e.g. Noida Toll
  Bridge surged ~166% in 2023-2024 on litigation speculation while its
  loss per share actually WIDENED from -1.70 to -13.11 over the same
  window, and independent quality screens rated it poor; it later gave
  back the vast majority of that move
- Revenue growth off a very small/depressed base that looks dramatic in
  percentage terms but isn't durable
- Poor or "not eligible" ratings on independent quality screens, combined
  with an "expensive valuation" flag despite weak or negative earnings
- This isn't just single-stock risk: the Nifty Smallcap 250 and Microcap
  250 indices both fell over 20% in a 2025 reversal, with roughly 79-80
  stocks in that universe individually down 30-50%, after "multibagger"
  narratives had become widespread. Survivorship bias is empirically what
  happened to a large share of the same-looking cohort, not a theoretical
  caveat.

## Mandatory: search for genuinely comparable sector precedents

Beyond the fixed examples above (which may not be in this stock's sector),
use web_search to find 1-3 real companies IN THE SAME OR AN ADJACENT
SECTOR that had a similar setup (similar catalyst type, similar growth/
re-rating profile) at some point in the past, and briefly note what
actually happened to them - whether the setup played out or failed. If you
cannot find a genuinely comparable sector precedent, say so explicitly
rather than stretching a loose comparison. A specific, sourced sector
precedent is far more useful here than a generic analogy.

Golden rules, non-negotiable, in every single report you produce:

1. NEVER state a future return, price, or timeline as a fact. Every
   forward statement is a labeled scenario or an estimated plausible
   range, never a prediction.
2. NEVER issue a Buy/Hold/Sell recommendation. You may state entry/target/
   invalidation NUMBERS in the Position Framework section, but every number
   there must be explicitly derived from and labeled against the Bear/Base/
   Best scenarios - never state them as a standalone instruction to act.
   Your job ends at showing what the stated framework produces; the
   decision is theirs.
3. The Shariah compliance screen is MANDATORY and runs in every report,
   regardless of what else is asked. Use standard AAOIFI-style thresholds:
   - Sector screen (no conventional banking/insurance, alcohol, gambling,
     pork, conventional weapons, adult entertainment)
   - Debt ratio: interest-bearing debt / market cap, should be < 33.33%
   - Cash ratio: (cash + interest-bearing investments) / market cap,
     should be < 33.33%
   - Impure income ratio: (interest income + non-permissible revenue) /
     total revenue, should be < 5.00%
   State a verdict of PASS, FAIL, or UNCERTAIN (use UNCERTAIN when a ratio
   is genuinely close to its threshold or data is incomplete).
4. A {TARGET_RETURN_PCT}% move on a SINGLE STOCK is a materially aggressive
   target regardless of how much time is allowed for it - reality-check
   this explicitly before giving any scenario detail.
5. Never quote copyrighted text at length. Paraphrase in your own words;
   at most one short quote under 15 words per source if genuinely needed.

Structure your report with these sections, using Markdown:

## Snapshot
Company, ticker, exchange, current price, market cap, one-line business
description. Lead with any serious governance/safety red flag found.
Cite the real computed market data (trend, RSI, volatility) briefly here.

## Return Target Reality Check
Per golden rule 4.

## Quality of Earnings & Governance
Cash conversion, debt trend, promoter/insider holding and pledging,
litigation, auditor history, related-party exposure. Also search for
recent bulk/block deal disclosures and insider trading filings (large
promoter or institutional buying is a real positive signal; large selling
a real caution flag) - this is standard practice for Indian equities
specifically and easy to miss if not explicitly checked.

## Analyst Consensus & Structural Catalysts
Not daily news - search for current sell-side consensus rating/target and
its recent trend if findable, stated as their view, not yours, and never
a substitute for your own scenario modeling below. Separately, note any
genuinely STRUCTURAL catalyst (a policy/regulatory shift, a durable
multi-year contract, a secular sector shift) distinct from routine
quarterly news, which belongs in Catalysts & Risks instead.

## Shariah Compliance Screen
Per golden rule 3, showing actual ratio numbers.

## Pattern Match: Genuine Setup or Look-Alike?
Compare this stock's own growth trend, ROE/ROCE, ownership/liquidity, and
catalyst against the two patterns above, AND against the sector-specific
comparable precedent(s) you searched for. Say which it resembles more, or
that it's ambiguous.

## Valuation Snapshot
Current multiple vs. own history and vs. 2-4 real peers if findable. Also
report the real computed sector-relative strength figures (this stock's
return vs. its own sector benchmark, not just the broad index) - a stock
outperforming its sector specifically is a different, more targeted
signal than just outperforming the broad market.

## Scenario Modeling: Bear / Base / Best
Address plausibility and ESTIMATED TIMELINE to reach a {TARGET_RETURN_PCT}%
move. Ground the Best case against the sector precedent if one applies.

## Position Framework: Entry, Downside Invalidation, and a {INVESTMENT_AMOUNT}
Example Position
This section gives concrete numbers, but they must be DERIVED from the
scenario modeling above, not a separate confident instruction. Never write
this as "buy at X, sell at Y" - frame every number as what a specific
stated method produces, tied explicitly to the Bear/Base/Best scenarios
already built. Use the stock's own listing currency symbol throughout
(₹ for an NSE/BSE listing, $ for a US listing, etc.) - {INVESTMENT_AMOUNT}
is a raw number, apply the correct currency to it yourself based on where
this stock trades.
- **Entry consideration**: state the current price as the reference entry
  point (this is a long-horizon thesis, not a timed technical entry) - note
  briefly, using the real computed RSI/trend data given to you, whether the
  stock currently looks technically extended or currently depressed
  relative to its own recent range, since that affects how much weight to
  put on entering right now vs. waiting, but do not give a target entry
  price different from CMP.
- **Downside invalidation level**: NOT an arbitrary stop-loss - state the
  Bear-case price level already computed above as the reference point:
  "a decline toward the Bear-case level of X would suggest the thesis
  is playing out unfavorably and is worth revisiting," plus, if the quality/
  governance section flagged a specific real concern (e.g. rising debtor
  days, promoter pledging), name that as a fundamental invalidation
  trigger too, not just a price level.
- **Targets**: restate the Base and Best case price targets from the
  scenario section above as the reference exit levels for this framework -
  do not invent new target numbers here.
- **{INVESTMENT_AMOUNT} illustration**: using the current price, compute
  roughly how many shares/units {INVESTMENT_AMOUNT} (in the stock's own
  listing currency) would buy, then show what that position would be worth
  at the Bear, Base, and Best case price targets - as three separate
  scenario outcomes, each explicitly labeled as illustrative and NOT a
  prediction of which one will occur. Show the absolute gain/loss and the
  percentage for each. State plainly this is arithmetic applied to the
  scenarios above, not a forecast of actual results, and that real returns
  depend on the actual entry/exit prices and timing the person executes at,
  which will differ from this illustration.

## Catalysts & Risks
2-3 sourced upside catalysts, 2-3 sourced downside risks.

## Closing Comparison
One paragraph: plausibility and rough timeline, no Buy/Hold/Sell call.
Note real data-quality limitations actually encountered this run.

After the full report, end with a fenced ```json block (nothing after it):

```json
{{
  "ticker": "...",
  "company_name": "...",
  "exchange": "...",
  "current_price": "...",
  "market_cap": "...",
  "currency_symbol": "\u20b9 or $ or whatever this stock's listing currency is",
  "entry_price": "plain number, no currency symbol or commas, e.g. 1254.90",
  "downside_level_price": "plain number, the Bear-case/invalidation level",
  "target_base_price": "plain number, the Base case target",
  "target_best_price": "plain number, the Best case target",
  "shariah_verdict": "PASS | FAIL | UNCERTAIN",
  "pattern_match": "GENUINE | LOOK_ALIKE | AMBIGUOUS",
  "target_plausibility": "PLAUSIBLE_FAST | PLAUSIBLE_SLOW | UNLIKELY",
  "one_line_summary": "..."
}}
```

The entry/downside/target numeric fields MUST exactly match the figures
you stated in prose in the Position Framework section above - these drive
an independently-computed verification table, so if they don't match your
own prose, the published report will visibly disagree with itself.
"""


def _to_cacheable_blocks(content_blocks) -> list:
    """Convert SDK response content blocks to plain dicts and mark the last
    one as a cache breakpoint, so the NEXT call in this loop gets a cache
    hit (~90% cheaper) on everything up to and including this turn, instead
    of being billed full price for resending the whole growing history."""
    blocks = []
    for b in content_blocks:
        blocks.append(b.model_dump() if hasattr(b, "model_dump") else dict(b))
    if blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def run_analysis(query: str, computed_stats_block: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_message = f"Analyze this stock: {query}\n\n{computed_stats_block}"
    messages = [{"role": "user", "content": user_message}]
    full_text_parts = []
    usage_totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0}

    # System prompt is long (~2-3K tokens) and identical on every call in
    # this loop - caching it means only the first call pays full price for
    # it; every continuation reads it at ~10% of that cost instead.
    system_blocks = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    for _ in range(MAX_CONTINUATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": SEARCH_MAX_USES}],
            messages=messages,
        )
        usage_totals["calls"] += 1
        usage_totals["input"] += getattr(response.usage, "input_tokens", 0) or 0
        usage_totals["output"] += getattr(response.usage, "output_tokens", 0) or 0
        usage_totals["cache_read"] += getattr(response.usage, "cache_read_input_tokens", 0) or 0
        usage_totals["cache_write"] += getattr(response.usage, "cache_creation_input_tokens", 0) or 0

        for block in response.content:
            if getattr(block, "type", None) == "text":
                full_text_parts.append(block.text)
        if response.stop_reason == "pause_turn":
            # Mark this turn as a cache breakpoint before continuing, so
            # the next call reads this growing prefix cheaply instead of
            # paying full input price for the whole accumulated context.
            messages.append({"role": "assistant", "content": _to_cacheable_blocks(response.content)})
            continue
        break

    print(f"[usage] calls={usage_totals['calls']} input={usage_totals['input']} "
          f"output={usage_totals['output']} cache_read={usage_totals['cache_read']} "
          f"cache_write={usage_totals['cache_write']}")

    return "\n".join(full_text_parts)


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_long_term.py <ticker or company name>")
        sys.exit(1)

    query = sys.argv[1]
    force_rerun = os.environ.get("FORCE_RERUN", "false").lower() == "true"
    dup_window_days = int(os.environ.get("DUP_GUARD_DAYS", "3"))

    slug_check = report_utils.slugify(query)
    if not force_rerun:
        existing = report_utils.find_recent_entry(slug_check, HORIZON_SUBDIR, dup_window_days)
        if existing:
            print(f"[cost-guard] A long-term report for '{query}' already exists from "
                  f"{existing.get('date')} (within the last {dup_window_days} days) at "
                  f"docs/reports/{HORIZON_SUBDIR}/{existing.get('slug')}.html")
            print("[cost-guard] Skipping the API call to avoid a duplicate charge. "
                  "Set FORCE_RERUN=true (or the workflow's force_rerun input) to re-run anyway.")
            return

    print(f"[long-term] Fetching real market data for: {query}")
    stats = market_data.fetch_and_compute(query)
    stats_block = market_data.format_for_prompt(stats)
    print(stats_block)

    print(f"[long-term] Running analysis for: {query}")
    report_text = run_analysis(query, stats_block)
    meta, body_md = report_utils.extract_metadata(report_text)

    if meta is None:
        print("Warning: model did not return parseable metadata JSON; using fallback values.")
        meta = {"ticker": query, "company_name": query, "shariah_verdict": "UNCERTAIN",
                 "pattern_match": "AMBIGUOUS", "target_plausibility": "PLAUSIBLE_SLOW",
                 "one_line_summary": "See full report."}

    date_str = datetime.date.today().isoformat()
    slug = report_utils.slugify(meta.get("ticker") or query)
    title = meta.get("company_name") or meta.get("ticker") or "Report"

    shariah = meta.get("shariah_verdict", "UNCERTAIN").upper()
    pattern = meta.get("pattern_match", "AMBIGUOUS").upper()
    target = meta.get("target_plausibility", "PLAUSIBLE_SLOW").upper()

    badge_defs = [
        (f"Shariah: {shariah}", {"PASS": "pass", "FAIL": "fail"}.get(shariah, "uncertain")),
        ({"GENUINE": "Resembles genuine setups", "LOOK_ALIKE": "Resembles look-alike traps"}
         .get(pattern, "Pattern unclear"), {"GENUINE": "pass", "LOOK_ALIKE": "fail"}.get(pattern, "uncertain")),
        ({"PLAUSIBLE_FAST": f"Plausible within ~{MIN_HOLD_MONTHS}-18mo",
          "PLAUSIBLE_SLOW": "Plausible, likely multi-year",
          "UNLIKELY": "Unlikely without unusual outcome"}.get(target, target),
         {"PLAUSIBLE_FAST": "pass", "UNLIKELY": "fail"}.get(target, "uncertain")),
    ]

    hero_stat_defs = [
        (meta.get("current_price"), "Price", False),
        (meta.get("market_cap"), "Market Cap", True),
        (meta.get("exchange"), "Exchange", False),
    ]

    currency_symbol = meta.get("currency_symbol", "")
    verified_table = report_utils.compute_verified_position_table(meta, INVESTMENT_AMOUNT, currency_symbol)

    raw_ohlcv = stats.get("_raw_ohlcv")
    backtest_html = ""
    if raw_ohlcv:
        try:
            bt_stats = backtest.backtest_long_term_holding(raw_ohlcv, TARGET_RETURN_PCT, MIN_HOLD_MONTHS)
            backtest_html = backtest.format_long_term_backtest_html(bt_stats, TARGET_RETURN_PCT, MIN_HOLD_MONTHS)
            print(f"[backtest] long-term: {bt_stats}")
        except Exception as e:  # noqa: BLE001 - a backtest failure must never break the whole report
            print(f"[backtest] skipped due to error: {e}")

    html_out = report_utils.build_report_html(title, badge_defs, body_md, date_str, hero_stat_defs,
                                               verified_table + backtest_html)
    path = report_utils.write_report(HORIZON_SUBDIR, slug, html_out)
    print(f"Wrote report: {path}")

    report_utils.save_entry(
        {
            "ticker": meta.get("ticker", query),
            "company_name": title,
            "one_line_summary": meta.get("one_line_summary", ""),
            "date": date_str,
            "horizon": "long",
            "index_row": [(b[0], b[1]) for b in badge_defs],
        },
        HORIZON_SUBDIR,
        slug,
    )
    print("Updated docs/index.html")


if __name__ == "__main__":
    main()
