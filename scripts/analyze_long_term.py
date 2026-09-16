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

TARGET_RETURN_PCT = float(os.environ.get("TARGET_RETURN_PCT", "50"))
MIN_HOLD_MONTHS = int(os.environ.get("MIN_HOLD_MONTHS", "12"))
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = 8000
MAX_CONTINUATIONS = 6
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
DATA (moving averages, RSI, volatility, drawdown, momentum, beta) that was
calculated directly from actual price history in Python before this
conversation started - not sourced from a website, not estimated by you.
Treat these as ground truth. Use your web_search tool for everything else:
financials, governance, news, and - importantly - genuinely comparable
sector precedents (see below).

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
2. NEVER issue a Buy/Hold/Sell recommendation. Your job ends at describing
   how plausible the {TARGET_RETURN_PCT}% target looks and roughly what
   timeline comparable real cases took - the decision is theirs.
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
litigation, auditor history, related-party exposure.

## Shariah Compliance Screen
Per golden rule 3, showing actual ratio numbers.

## Pattern Match: Genuine Setup or Look-Alike?
Compare this stock's own growth trend, ROE/ROCE, ownership/liquidity, and
catalyst against the two patterns above, AND against the sector-specific
comparable precedent(s) you searched for. Say which it resembles more, or
that it's ambiguous.

## Valuation Snapshot
Current multiple vs. own history and vs. 2-4 real peers if findable.

## Scenario Modeling: Bear / Base / Best
Address plausibility and ESTIMATED TIMELINE to reach a {TARGET_RETURN_PCT}%
move. Ground the Best case against the sector precedent if one applies.

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
  "shariah_verdict": "PASS | FAIL | UNCERTAIN",
  "pattern_match": "GENUINE | LOOK_ALIKE | AMBIGUOUS",
  "target_plausibility": "PLAUSIBLE_FAST | PLAUSIBLE_SLOW | UNLIKELY",
  "one_line_summary": "..."
}}
```
"""


def run_analysis(query: str, computed_stats_block: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_message = f"Analyze this stock: {query}\n\n{computed_stats_block}"
    messages = [{"role": "user", "content": user_message}]
    full_text_parts = []

    for _ in range(MAX_CONTINUATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 15}],
            messages=messages,
        )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                full_text_parts.append(block.text)
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        break

    return "\n".join(full_text_parts)


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_long_term.py <ticker or company name>")
        sys.exit(1)

    query = sys.argv[1]
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

    html_out = report_utils.build_report_html(title, badge_defs, body_md, date_str)
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
