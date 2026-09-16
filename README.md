# Stock Analysis Agents

Two automated research agents, sharing one GitHub Pages dashboard:

- **Long-term** (`analyze_long_term.py`, workflow: "Analyze Stock (Long-term,
  12mo+ hold)"): hold at least a minimum number of months, sell the first
  time unrealized return reaches your target — whenever that happens,
  ideally sooner. Fundamentals-driven: quality of earnings, governance,
  full Shariah ratio screen, valuation, and a pattern match against real
  historical cases of genuine multi-year re-ratings vs. look-alike traps.
- **Short-term** (`analyze_short_term.py`, workflow: "Analyze Stock
  (Short-term, <12mo)"): a different question — not "is this a good
  business" but "is there a real, technically-confirmed setup with a
  near-term catalyst." Technical-driven: trend/momentum/volume alignment,
  a specific dated catalyst (or the honest absence of one), India-specific
  execution risk (circuit filters, liquidity, event gap risk), and a
  lighter/faster Shariah check appropriate for the shorter horizon.

They're kept separate on purpose — mixing a multi-year fundamental thesis
and a technical near-term setup into one prompt produces a worse, muddled
version of both. What they share: the GitHub Pages dashboard, the report
rendering, and — this is the important part — a **real computed market
data module**.

## Why "real computed" matters

Earlier versions of this relied on the model reading whatever RSI, beta,
or volatility figure a web search happened to surface from some finance
site — a pre-computed, possibly-stale, possibly-inconsistent number. That
isn't genuine quant analysis. `scripts/market_data.py` now fetches actual
historical price/volume data (via `yfinance`) and computes, directly in
Python, before the model ever sees the stock:

- 50-day / 200-day moving averages and a trend read
- RSI(14) and a MACD signal-cross read
- Annualized volatility and maximum drawdown
- 3/6/12-month momentum
- Recent volume vs. 20-day average
- Beta against a relevant benchmark (Nifty 50 for `.NS`/`.BO` tickers,
  S&P 500 otherwise), computed via real regression against actual
  historical returns — not looked up

These are injected into the prompt as labeled ground truth. Web search is
reserved for what it's actually good for: financials, governance, news,
and sourced sector-comparable precedents.

**Honesty about testing**: the pure math in `market_data.py` (RSI, MACD,
moving averages, volatility, drawdown, momentum, beta regression) was unit
-tested against synthetic price data before this was packaged. The actual
live network call to Yahoo Finance could **not** be tested end-to-end in
the sandbox this was built in (no outbound access to finance data sites
there) — it should work in GitHub Actions' runner, which has full internet
access, but verify on your first real run and check the Action's logs if
a report comes back saying computed market data was unavailable.

## How it actually works

GitHub Pages only serves static files — it can't run the agents
themselves. The real work happens in **GitHub Actions**, triggered
manually by you entering a ticker. Each workflow fetches real market data,
calls the Claude API, writes the finished report as HTML into
`docs/reports/long/` or `docs/reports/short/`, and commits it — which
Pages then serves via one combined dashboard (`docs/index.html`) showing
both horizons together, so you can browse and compare both kinds of
report.

## Setup (one-time)

1. **Create a GitHub repository** and add all files in this project.

2. **Get an Anthropic API key** from console.anthropic.com if you don't
   have one. Paid API — each run costs real money (more for the long-term
   agent, which does more searching; a few cents to low dollars per run).

3. **Add it as a repository secret**: Settings → Secrets and variables →
   Actions → New repository secret → name `ANTHROPIC_API_KEY`. Never
   commit the key into any file.

4. **Enable GitHub Pages**: Settings → Pages → Source: "Deploy from a
   branch" → Branch: `main`, folder: **`/docs`**.

5. **Run an analysis**: Actions tab → pick either workflow → "Run
   workflow" → enter a ticker (and optionally adjust the target % / hold
   window) → wait a few minutes → refresh your Pages URL.

## What each agent checks, every time

**Both:**
- A reality check on the target return, stated plainly, before any numbers
- A mandatory Shariah compliance check (full AAOIFI-style ratios for
  long-term; a lighter, faster version for short-term, explicitly labeled
  as such)
- A pattern match against real historical cases — genuine setups vs.
  look-alike/speculative traps — plus, for the long-term agent, an
  explicit web search for a genuinely comparable precedent in the same
  sector, not just the fixed examples baked into the prompt

**Long-term only:** quality of earnings/governance, valuation vs. peers,
Bear/Base/Best scenarios framed as plausibility + estimated timeline.

**Short-term only:** trend/momentum/volume alignment (from real computed
data), a specific dated near-term catalyst or its honest absence,
India-specific execution risk (circuit filters, liquidity/surveillance
stage, event-gap risk inside the hold window).

## What's covered vs. what's still a real gap

Being direct about this rather than implying full coverage:

- **Fundamental** — solid (long-term agent).
- **Technical** — solid (short-term agent), now backed by real computed
  indicators rather than searched figures.
- **Quant** — genuinely improved: real volatility, drawdown, momentum, and
  regression-based beta, computed from actual price history.
- **Sentiment** — present but qualitative (the model reads and reasons
  over search results); not a systematic, scored sentiment pipeline across
  many articles. A concrete upgrade path: add a step that pulls N recent
  headlines and has the model score each individually before aggregating,
  rather than reasoning over search results in one pass.
- **Time series / forecasting** — still limited to the computed statistics
  above (trend, momentum, volatility), not formal statistical forecasting
  (e.g., ARIMA, seasonality decomposition). Scenario modeling here is
  reasoned/qualitative, grounded in real numbers, not a statistical
  forecast model — worth knowing the difference.

## Customizing

- `scripts/market_data.py` — the computed-stats layer; add more indicators
  here if you want them (e.g. Bollinger Bands, ATR for the short-term
  agent's risk sizing).
- `scripts/analyze_long_term.py` / `analyze_short_term.py` — the actual
  agent logic and system prompts.
- `scripts/report_utils.py` — shared HTML rendering and the combined index.
- `docs/assets/style.css` — appearance.

## Known limitations, honestly stated

- The Markdown-to-HTML conversion is minimal and dependency-free. A real
  Markdown library is a straightforward upgrade if you want more fidelity.
- Each run is independent — no memory of past runs on the same stock
  beyond what's found via fresh web search.
- Web search results and company data are only as current and accurate as
  what the model finds at run time — treat every report as a starting
  point for your own research, not a final answer.
- This produces analytical framework output, not personalized investment
  advice, and nobody involved in building it is a licensed financial
  advisor.
