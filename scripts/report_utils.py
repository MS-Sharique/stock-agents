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
<header class="site-header">
  <a href="../../index.html" class="back-link">&larr; All reports</a>
</header>
<main class="report">
  <div class="report-meta">
    {badges}
    <span class="report-date">Generated {date}</span>
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


def build_report_html(title: str, badge_defs: list, body_md: str, date_str: str) -> str:
    body_html = markdown_to_html_basic(body_md)
    return REPORT_TEMPLATE.format(
        title=html.escape(title),
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
    rows = []
    for e in sorted(all_entries, key=lambda x: x.get("date", ""), reverse=True):
        horizon_label = "Long-term (12mo+)" if e.get("horizon") == "long" else "Short-term (<12mo)"
        horizon_class = "pass" if e.get("horizon") == "long" else "uncertain"
        badge_html = "".join(
            f'<span class="badge badge-{BADGE_CLASS.get(ck,"uncertain")}">{html.escape(lbl)}</span> '
            for lbl, ck in e.get("index_row", [])
        )
        link = f"reports/{e['horizon_subdir']}/{e['slug']}.html"
        rows.append(f"""
        <tr>
          <td><a href="{link}">{html.escape(e.get('company_name') or e.get('ticker',''))}</a></td>
          <td>{html.escape(e.get('ticker',''))}</td>
          <td><span class="badge badge-{horizon_class}">{horizon_label}</span></td>
          <td>{badge_html}</td>
          <td>{html.escape(e.get('one_line_summary',''))}</td>
          <td>{e.get('date','')}</td>
        </tr>""")

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
