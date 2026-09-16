"""
Shared rendering: turns a model's Markdown report + metadata into an HTML
report page, and maintains a combined index across both the long-term and
short-term agents so the person can browse both in one dashboard.
"""

import html
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(REPO_ROOT, "docs", "reports")
INDEX_PATH = os.path.join(REPO_ROOT, "docs", "index.html")
INDEX_DATA_PATH = os.path.join(REPO_ROOT, "docs", "reports", "index_data.json")

BADGE_CLASS = {"pass": "pass", "fail": "fail", "uncertain": "uncertain"}


def _parse_number(value) -> float:
    """Strip currency symbols/commas/whitespace and parse a number the
    model wrote in its JSON metadata. Raises ValueError if unparseable -
    callers should catch this and skip the verified table rather than
    crash the whole run over a formatting slip."""
    if value is None:
        raise ValueError("missing value")
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    if not cleaned or cleaned in ("-", "."):
        raise ValueError(f"unparseable: {value!r}")
    return float(cleaned)


def compute_verified_position_table(meta: dict, investment_amount: str, currency_symbol: str = "") -> str:
    """Independently recompute the investment illustration from the raw
    numeric fields the model reported (entry_price, downside_level_price,
    target_base_price, target_best_price), rather than trusting the
    model's own in-prose arithmetic. Returns an HTML snippet, or an empty
    string if the required fields aren't present/parseable - this must
    never crash the run over a missing field."""
    try:
        entry = _parse_number(meta.get("entry_price"))
        amount = _parse_number(investment_amount)
        if entry <= 0 or amount <= 0:
            raise ValueError("non-positive entry or amount")
        units = amount / entry

        rows = []
        for label, key in [
            ("Downside level", "downside_level_price"),
            ("Base case target", "target_base_price"),
            ("Best case target", "target_best_price"),
        ]:
            raw = meta.get(key)
            if raw is None:
                continue
            try:
                price = _parse_number(raw)
            except ValueError:
                continue
            value = units * price
            change = value - amount
            pct = (change / amount) * 100
            sign = "+" if change >= 0 else "-"
            rows.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td>{currency_symbol}{price:,.2f}</td>"
                f"<td>{currency_symbol}{value:,.2f}</td>"
                f"<td>{sign}{currency_symbol}{abs(change):,.2f} ({sign}{abs(pct):.1f}%)</td></tr>"
            )

        if not rows:
            return ""

        return f"""
<h3>Recomputed Arithmetic (Python, not the model's own math)</h3>
<p>Using an entry of {currency_symbol}{entry:,.2f}, {currency_symbol}{amount:,.0f}
buys approximately {units:,.2f} shares/units (a flat amount, not adjusted
for this stock's volatility - see the risk-adjusted sizing below if this
report includes it). <strong>This only verifies the
arithmetic</strong> - that units × price is calculated correctly - it does
NOT verify that the entry/downside/target prices themselves are accurate
or well-derived; those still come entirely from the model's own analysis
above and carry the same uncertainty as everything else in this report.
If this table disagrees with the model's prose on the MATH given the same
prices, trust this table. If you doubt the prices themselves, that's a
different question this table can't answer.</p>
<table class="report-table">
<tr><th>Scenario</th><th>Price</th><th>Position Value</th><th>Change</th></tr>
{''.join(rows)}
</table>
"""
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"[verified-table] skipped: {e}")
        return ""


def compute_risk_adjusted_sizing(meta: dict, capital_amount: str, risk_pct_per_trade: float,
                                  currency_symbol: str = "") -> str:
    """Standard risk-based position sizing: risk a fixed % of capital per
    trade, size the position from the stop distance, rather than putting a
    flat dollar amount in regardless of how wide the stop is. This is the
    same method from the swing-trading skill's risk_management.md - never
    carried into this automated agent until now.

    Only meaningful when a real stop distance exists (short-term agent,
    ATR-based). Returns '' gracefully if fields are missing."""
    try:
        entry = _parse_number(meta.get("entry_price"))
        stop = _parse_number(meta.get("downside_level_price"))
        capital = _parse_number(capital_amount)
        stop_distance = entry - stop
        if stop_distance <= 0 or capital <= 0:
            raise ValueError("non-positive stop distance or capital")

        risk_amount = capital * (risk_pct_per_trade / 100)
        risk_based_shares = risk_amount / stop_distance
        risk_based_position_value = risk_based_shares * entry
        flat_shares = capital / entry

        capped_note = ""
        if risk_based_position_value > capital:
            capped_note = (f" This exceeds your stated {currency_symbol}{capital:,.0f} capital, "
                            f"meaning a {risk_pct_per_trade:.1f}% risk on this wide a stop would "
                            f"require more capital than stated - a signal that either the stop is "
                            f"too wide for this account size, or the risk % should be lower for "
                            f"this specific trade.")
            risk_based_shares = flat_shares  # cap at what capital actually allows
            risk_based_position_value = capital

        return f"""
<h3>Risk-Adjusted Position Sizing (standard method: risk a fixed % of capital, not a flat amount)</h3>
<p>Standard practice sizes a position from how far away the stop is, not
from a flat dollar amount - a wide stop should mean a SMALLER position for
the same dollar risk, not the same position size regardless. Risking
<strong>{risk_pct_per_trade:.1f}%</strong> of {currency_symbol}{capital:,.0f}
capital ({currency_symbol}{risk_amount:,.2f}) against a stop distance of
{currency_symbol}{stop_distance:,.2f} ({currency_symbol}{entry:,.2f} entry
to {currency_symbol}{stop:,.2f} stop) gives:</p>
<table class="report-table">
<tr><th></th><th>Shares/units</th><th>Position value</th></tr>
<tr><td>Flat amount (all {currency_symbol}{capital:,.0f} deployed)</td><td>{flat_shares:,.2f}</td><td>{currency_symbol}{capital:,.2f}</td></tr>
<tr><td>Risk-adjusted ({risk_pct_per_trade:.1f}% risk)</td><td>{risk_based_shares:,.2f}</td><td>{currency_symbol}{risk_based_position_value:,.2f}</td></tr>
</table>
<p>{capped_note} This is a sizing convention, not a recommendation of how
much of your capital to allocate to any single position - that's a
broader portfolio decision this report doesn't have visibility into.</p>
"""
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"[risk-sizing] skipped: {e}")
        return ""


def extract_metadata(report_text: str):
    match = re.search(r"```json\s*(\{.*?\})\s*```", report_text, re.DOTALL)
    if not match:
        return None, report_text
    raw = match.group(1)
    body = report_text[: match.start()].rstrip()
    try:
        return json.loads(raw), body
    except json.JSONDecodeError:
        return None, body


def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


def markdown_to_html_basic(md_text: str) -> str:
    """Minimal, dependency-free Markdown -> HTML for headings/paragraphs/
    bold/lists/tables. Swap for a real Markdown library if you want more
    fidelity - noted in the README."""
    lines = md_text.split("\n")
    out = []
    in_list = False
    in_table = False
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h2>{html.escape(stripped[3:])}</h2>")
            continue
        if stripped.startswith("### "):
            out.append(f"<h3>{html.escape(stripped[4:])}</h3>")
            continue
        if stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline_md(stripped[2:])}</li>")
            continue
        else:
            if in_list:
                out.append("</ul>")
                in_list = False

        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not in_table:
                out.append('<table class="report-table">')
                in_table = True
                out.append("<tr>" + "".join(f"<th>{inline_md(c)}</th>" for c in cells) + "</tr>")
                continue
            if set(stripped.replace("|", "").replace(":", "").strip()) <= {"-"}:
                continue
            out.append("<tr>" + "".join(f"<td>{inline_md(c)}</td>" for c in cells) + "</tr>")
            continue
        else:
            if in_table:
                out.append("</table>")
                in_table = False

        if stripped == "":
            out.append("")
            continue

        out.append(f"<p>{inline_md(stripped)}</p>")

    if in_list:
        out.append("</ul>")
    if in_table:
        out.append("</table>")

    return "\n".join(out)


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - Stock Analysis</title>
<link rel="stylesheet" href="../../assets/style.css">
</head>
<body>
<a href="../../index.html" class="back-link">&larr; All reports</a>
<main class="report">
  <div class="report-hero">
    <h1>{title}</h1>
    <div class="hero-stats">
      {hero_stats}
    </div>
    <div class="report-meta">
      {badges}
      <span class="report-date">Generated {date}</span>
    </div>
  </div>
  {body}
</main>
<footer class="site-footer">
  <p>Automated research output. Not personalized investment advice. Every
  forward figure above is a labeled scenario, not a prediction.</p>
</footer>
</body>
</html>
"""


def render_badges(badge_defs: list) -> str:
    """badge_defs: list of (label_text, class_key) tuples, class_key one of
    pass/fail/uncertain."""
    parts = []
    for label, class_key in badge_defs:
        css = BADGE_CLASS.get(class_key, "uncertain")
        parts.append(f'<span class="badge badge-{css}">{html.escape(label)}</span>')
    return "\n    ".join(parts)


def render_hero_stats(stat_defs: list) -> str:
    """stat_defs: list of (value_text, label_text, is_gold) tuples."""
    parts = []
    for value, label, is_gold in stat_defs:
        if not value:
            continue
        gold_class = " gold" if is_gold else ""
        parts.append(
            f'<div><div class="hero-stat-value{gold_class}">{html.escape(str(value))}</div>'
            f'<div class="hero-stat-label">{html.escape(label)}</div></div>'
        )
    return "\n      ".join(parts)


def build_report_html(title: str, badge_defs: list, body_md: str, date_str: str,
                       hero_stat_defs: list = None, verified_table_html: str = "") -> str:
    body_html = markdown_to_html_basic(body_md)
    if verified_table_html:
        body_html += verified_table_html
    hero_stats_html = render_hero_stats(hero_stat_defs or [])
    return REPORT_TEMPLATE.format(
        title=html.escape(title),
        hero_stats=hero_stats_html,
        badges=render_badges(badge_defs),
        date=date_str,
        body=body_html,
    )


def write_report(horizon_subdir: str, slug: str, html_content: str) -> str:
    out_dir = os.path.join(REPORTS_DIR, horizon_subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{slug}.html")
    with open(path, "w") as f:
        f.write(html_content)
    return path


def load_index_data() -> list:
    if os.path.exists(INDEX_DATA_PATH):
        with open(INDEX_DATA_PATH) as f:
            return json.load(f)
    return []


def find_recent_entry(slug: str, horizon_subdir: str, within_days: int):
    """Cost guard: check whether a report for this exact slug/horizon
    already exists from within the last N days, so a re-run doesn't
    silently double-charge for a report that's still fresh."""
    import datetime as _dt
    cutoff = _dt.date.today() - _dt.timedelta(days=within_days)
    for e in load_index_data():
        if e.get("slug") == slug and e.get("horizon_subdir") == horizon_subdir:
            try:
                entry_date = _dt.date.fromisoformat(e.get("date", ""))
            except ValueError:
                continue
            if entry_date >= cutoff:
                return e
    return None


def save_entry(entry: dict, horizon_subdir: str, slug: str):
    """entry must include: company_name/ticker, date, horizon ('long'/
    'short'), index_row (list of (label, class_key) for the index table's
    status column), one_line_summary."""
    all_entries = load_index_data()
    all_entries = [e for e in all_entries if not (e.get("slug") == slug and e.get("horizon_subdir") == horizon_subdir)]
    entry = dict(entry)
    entry["slug"] = slug
    entry["horizon_subdir"] = horizon_subdir
    all_entries.append(entry)
    with open(INDEX_DATA_PATH, "w") as f:
        json.dump(all_entries, f, indent=2)
    rebuild_index(all_entries)


def rebuild_index(all_entries: list):
    total = len(all_entries)
    shariah_pass = sum(1 for e in all_entries if e.get("shariah_verdict", "").upper() in ("PASS", "LIKELY_PASS"))
    genuine = sum(1 for e in all_entries if e.get("pattern_match", "").upper() == "GENUINE")
    long_count = sum(1 for e in all_entries if e.get("horizon") == "long")
    short_count = total - long_count

    rows = []
    for e in sorted(all_entries, key=lambda x: x.get("date", ""), reverse=True):
        horizon_key = "long" if e.get("horizon") == "long" else "short"
        horizon_label = "Long-term" if horizon_key == "long" else "Short-term"
        badge_html = "".join(
            f'<span class="badge badge-{BADGE_CLASS.get(ck,"uncertain")}">{html.escape(lbl)}</span> '
            for lbl, ck in e.get("index_row", [])
        )
        link = f"reports/{e['horizon_subdir']}/{e['slug']}.html"
        rows.append(f"""
        <tr>
          <td><a href="{link}">{html.escape(e.get('company_name') or e.get('ticker',''))}</a></td>
          <td>{html.escape(e.get('ticker',''))}</td>
          <td><span class="horizon-tag {horizon_key}">{horizon_label}</span></td>
          <td>{badge_html}</td>
          <td>{html.escape(e.get('one_line_summary',''))}</td>
          <td>{e.get('date','')}</td>
        </tr>""")

    stats_strip = f"""
  <div class="stat-block">
    <div class="stat-value">{total}</div>
    <div class="stat-label">Reports generated</div>
  </div>
  <div class="stat-block">
    <div class="stat-value">{shariah_pass}/{total if total else 0}</div>
    <div class="stat-label">Shariah pass or likely-pass</div>
  </div>
  <div class="stat-block">
    <div class="stat-value gold">{genuine}/{total if total else 0}</div>
    <div class="stat-label">Genuine-pattern matches</div>
  </div>
  <div class="stat-block">
    <div class="stat-value">{long_count} / {short_count}</div>
    <div class="stat-label">Long-term / short-term</div>
  </div>""" if total else ""

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Analysis Agents</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<header class="site-header">
  <h1>Stock Analysis Agents</h1>
  <p class="subhead">Two agents, run from the Actions tab: a long-term one
  (hold 12mo+, sell at your target return whenever it's hit) and a
  short-term one (technical setup, sub-12-month). Both run a mandatory
  Shariah compliance check and reason over real computed market
  statistics, not just web-searched figures.</p>
</header>
<div class="stats-strip">{stats_strip}</div>
<main>
  <table class="index-table">
    <tr><th>Company</th><th>Ticker</th><th>Horizon</th><th>Status</th><th>Summary</th><th>Date</th></tr>
    {''.join(rows) if rows else '<tr><td colspan="6">No reports yet. Run a workflow to analyze a stock.</td></tr>'}
  </table>
</main>
<footer class="site-footer">
  <p>Automated research output. Not personalized investment advice.</p>
</footer>
</body>
</html>
"""
    with open(INDEX_PATH, "w") as f:
        f.write(index_html)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
