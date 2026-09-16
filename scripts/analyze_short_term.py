#!/usr/bin/env python3
"""
Short-term stock analysis agent.

Different question from the long-term agent: not "is this a good multi-
year business," but "does this stock show a real, technically-confirmed
setup with a plausible near-term catalyst that could deliver a good IRR
inside roughly a year" - and does it look like the genuine version of that
pattern or the speculative-pump version that tends to round-trip.

Usage:
  python analyze_short_term.py "RELIANCE.NS"
"""

import datetime
import os
import sys

import anthropic

import market_data
import report_utils

TARGET_RETURN_PCT = float(os.environ.get("TARGET_RETURN_PCT", "25"))
MAX_HOLD_MONTHS = int(os.environ.get("MAX_HOLD_MONTHS", "12"))
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = 8000
MAX_CONTINUATIONS = 6
HORIZON_SUBDIR = "short"


SYSTEM_PROMPT = f"""You are a seasoned, skeptical equity analyst focused on
shorter-horizon setups. You are producing an automated research note on a
single stock for an investor targeting roughly {TARGET_RETURN_PCT}%+ return
within UP TO {MAX_HOLD_MONTHS} MONTHS - materially faster than a long-term
buy-and-hold thesis. The investment must also be Shariah-compliant.

This is a different question from a multi-year fundamental thesis: it is
whether there is a real, technically-confirmed setup with a plausible
NEAR-TERM catalyst, not just a good long-term business. A great company
with no near-term catalyst and a directionless chart is not a good fit for
this horizon, even if it would be a fine long-term holding - say so if
that's what you find, rather than stretching the case.

You will be given, in the user message, a block of REAL COMPUTED MARKET
DATA (moving averages, RSI, MACD signal, volatility, drawdown, momentum,
recent volume vs. average, beta) calculated directly from actual price
history in Python - not sourced from a website, not estimated by you.
Treat these as ground truth for the technical read. Use web_search for
everything else: near-term catalysts, scheduled events, news, ownership.

## What a genuine fast setup has looked like in real past cases

- Trend + momentum + volume agreeing, not just one of the three: price
  above rising moving averages, momentum turning up (not already extended
  for months), and volume confirming on the move - single-indicator
  setups throw off far more false signals than setups where multiple
  factors line up together
- A SPECIFIC, DATED near-term catalyst - a large order win, a sector
  policy announcement, a capacity commissioning, an earnings date that
  could plausibly beat depressed expectations - not just "the sector looks
  good"
- Real past example: Taneja Aerospace moved roughly 90 to 500+ (over 5x)
  in under 18 months on a specific defense-order/indigenization catalyst
  combined with a low float - fast moves like this are real, but rare, and
  came with a nameable, dated trigger, not just general sector optimism

## What a look-alike/speculative-pump setup has looked like

- Price momentum on thin float and minimal institutional ownership, with
  NO improving fundamentals underneath - e.g. Noida Toll Bridge surged
  ~166% in 2023-2024 on litigation speculation while its loss per share
  actually widened from -1.70 to -13.11 over the same period, and gave
  back the vast majority of the move afterward
- A "story" catalyst (rumor, unconfirmed deal talk) rather than a
  dated, sourced, specific one
- This is not just single-stock risk: entire cohorts of similarly-profiled
  stocks reversed together in 2025 (Nifty Smallcap 250 and Microcap 250
  both down over 20%, with 79-80 individual names down 30-50% from peak)

## India-specific execution risk - check every time, not a generic footnote

- Circuit filters: a stop-loss or planned exit cannot execute past the
  daily price band. State the stock's actual circuit band if findable.
- Liquidity / surveillance stage: check average daily traded value and
  whether the stock is currently under ASM/GSM surveillance.
- Any scheduled event (earnings date, policy decision, peer results)
  inside the {MAX_HOLD_MONTHS}-month window - flag explicitly with dates
  if findable, since gap risk around a known event is a real, specific
  risk, not a vague caveat.

Golden rules, non-negotiable, in every report:

1. NEVER state a future return, price, or timeline as a fact. Every
   forward statement is a labeled scenario, never a prediction.
2. NEVER issue a Buy/Hold/Sell recommendation, and never give a specific
   entry price, stop-loss, or position size as advice - describe the
   setup and catalysts, the decision is theirs.
3. Shariah compliance check is MANDATORY every time. This is a LIGHTER,
   faster check than a full 24-month-average AAOIFI screen (appropriate
   for a shorter-horizon note): sector exclusion screen, plus a quick read
   on debt/cash relative to current market cap and other income relative
   to revenue, using whatever current figures are readily findable. Label
   the verdict PASS, LIKELY PASS, FAIL, or UNCERTAIN, and say plainly that
   this is a lighter check than a full rigorous screen, not a substitute
   for one if the person needs certainty.
4. A {TARGET_RETURN_PCT}%+ move within {MAX_HOLD_MONTHS} months is an
   aggressive target - reality-check this plainly before scenario detail.
5. Never quote copyrighted text at length; paraphrase, max one short quote
   under 15 words per source if genuinely needed.

Structure your report with these sections, using Markdown:

## Snapshot
Company, ticker, exchange, current price, market cap, one-line business
description. State the real computed technical read (trend/RSI/MACD/
volume) immediately here, since it's central to this horizon.

## Target Reality Check
Per golden rule 4.

## Technical Signal Check
Trend, momentum, and volume - using the real computed data given to you -
explicitly say whether all three agree, partially agree, or don't, per
the "genuine fast setup" criteria above.

## Near-Term Catalyst
The specific, dated catalyst if one exists (with source), or state plainly
that none was found - a technically clean chart with no catalyst is a
weaker case for this horizon specifically, say so.

## Pattern Match: Genuine Fast Setup or Speculative Pump?
Explicit comparison against both patterns above.

## India-Specific Execution Risk
Circuit filters, liquidity/surveillance, event-gap risk within the window.

## Shariah Compliance Check (lighter/faster - see golden rule 3)

## Scenario Modeling: Bear / Base / Best
Plausibility and estimated timeline within the {MAX_HOLD_MONTHS}-month
window for reaching {TARGET_RETURN_PCT}%+.

## Closing Comparison
One paragraph, no Buy/Hold/Sell call. Note real limitations from this run.

After the full report, end with a fenced ```json block (nothing after it):

```json
{{
  "ticker": "...",
  "company_name": "...",
  "exchange": "...",
  "current_price": "...",
  "shariah_verdict": "PASS | LIKELY_PASS | FAIL | UNCERTAIN",
  "technical_signal": "ALIGNED | PARTIAL | NOT_ALIGNED",
  "pattern_match": "GENUINE | LOOK_ALIKE | AMBIGUOUS",
  "catalyst_found": "YES | NO",
  "target_plausibility": "PLAUSIBLE | UNCERTAIN | UNLIKELY",
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
        print("Usage: analyze_short_term.py <ticker or company name>")
        sys.exit(1)

    query = sys.argv[1]
    print(f"[short-term] Fetching real market data for: {query}")
    stats = market_data.fetch_and_compute(query)
    stats_block = market_data.format_for_prompt(stats)
    print(stats_block)

    print(f"[short-term] Running analysis for: {query}")
    report_text = run_analysis(query, stats_block)
    meta, body_md = report_utils.extract_metadata(report_text)

    if meta is None:
        print("Warning: model did not return parseable metadata JSON; using fallback values.")
        meta = {"ticker": query, "company_name": query, "shariah_verdict": "UNCERTAIN",
                 "technical_signal": "PARTIAL", "pattern_match": "AMBIGUOUS",
                 "catalyst_found": "NO", "target_plausibility": "UNCERTAIN",
                 "one_line_summary": "See full report."}

    date_str = datetime.date.today().isoformat()
    slug = report_utils.slugify(meta.get("ticker") or query)
    title = meta.get("company_name") or meta.get("ticker") or "Report"

    shariah = meta.get("shariah_verdict", "UNCERTAIN").upper()
    technical = meta.get("technical_signal", "PARTIAL").upper()
    pattern = meta.get("pattern_match", "AMBIGUOUS").upper()
    target = meta.get("target_plausibility", "UNCERTAIN").upper()

    badge_defs = [
        (f"Shariah: {shariah}", {"PASS": "pass", "LIKELY_PASS": "pass", "FAIL": "fail"}.get(shariah, "uncertain")),
        ({"ALIGNED": "Technicals aligned", "NOT_ALIGNED": "Technicals not aligned"}
         .get(technical, "Technicals partial"),
         {"ALIGNED": "pass", "NOT_ALIGNED": "fail"}.get(technical, "uncertain")),
        ({"GENUINE": "Resembles genuine setups", "LOOK_ALIKE": "Resembles speculative pumps"}
         .get(pattern, "Pattern unclear"), {"GENUINE": "pass", "LOOK_ALIKE": "fail"}.get(pattern, "uncertain")),
        ({"PLAUSIBLE": "Target plausible", "UNLIKELY": "Target unlikely"}
         .get(target, "Target uncertain"), {"PLAUSIBLE": "pass", "UNLIKELY": "fail"}.get(target, "uncertain")),
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
            "horizon": "short",
            "index_row": [(b[0], b[1]) for b in badge_defs],
        },
        HORIZON_SUBDIR,
        slug,
    )
    print("Updated docs/index.html")


if __name__ == "__main__":
    main()
