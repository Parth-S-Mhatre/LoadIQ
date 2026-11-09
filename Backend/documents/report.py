"""Self-contained HTML report for the document MVP.

The report is deliberately dependency-free: it can be printed to PDF by the
browser or rendered by the React app later.  Charts are small inline SVG.
"""

from __future__ import annotations

from html import escape

from .schemas import AnalyticsResult, DocumentMeta, ExtractionResult


def _fmt(value: object, suffix: str = "") -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def _bar_chart(items: list[dict], currency: str) -> str:
    if not items:
        return "<p class='muted'>No cost breakdown available.</p>"
    max_value = max(float(item["value"]) for item in items) or 1.0
    rows = []
    for index, item in enumerate(items):
        width = max((float(item["value"]) / max_value) * 100.0, 2.0)
        rows.append(
            f"""
            <div class="bar-row">
              <span class="bar-label">{escape(str(item['name']))}</span>
              <span class="bar-track"><span class="bar-fill bar-{index % 4}" style="width:{width:.1f}%"></span></span>
              <span class="bar-value">{escape(currency)}{float(item['value']):,.2f}</span>
            </div>
            """
        )
    return "".join(rows)


def _usage_chart(rows: list[dict], currency: str) -> str:
    usable = [row for row in rows if row.get("consumption_kwh") is not None]
    if not usable:
        return "<p class='muted'>No usage history available yet.</p>"
    max_value = max(float(row["consumption_kwh"]) for row in usable) or 1.0
    bars = []
    for index, row in enumerate(usable):
        height = max((float(row["consumption_kwh"]) / max_value) * 120.0, 4.0)
        cost = row.get("total_cost")
        label = str(row.get("period") or "")
        bars.append(
            f"""
            <div class="usage-col" title="{escape(label)}: {float(row['consumption_kwh']):,.0f} kWh">
              <div class="usage-bar" style="height:{height:.1f}px"></div>
              <span class="usage-label">{escape(label[5:] if len(label) >= 10 else label)}</span>
            </div>
            """
        )
    return f"<div class='usage-chart'>{''.join(bars)}</div>"


def render_report(
    document: DocumentMeta,
    extraction: ExtractionResult,
    analytics: AnalyticsResult,
) -> str:
    summary = analytics.summary
    currency = str(summary.get("currency") or "£")
    insights_html = "".join(
        f"""
        <li class="insight {escape(insight.severity)}">
          <strong>{escape(insight.title)}</strong>
          <p>{escape(insight.detail)}</p>
          <small>Evidence: {escape(", ".join(insight.evidence_ids) or "—")}</small>
        </li>
        """
        for insight in analytics.insights
    ) or "<li>No insights generated.</li>"

    recommendations_html = "".join(
        f"""
        <li class="recommendation">
          <strong>{escape(rec.title)}</strong>
          <p>{escape(rec.detail)}</p>
          <small>
            Estimated savings: {escape(_fmt(rec.estimated_savings, ""))} {escape(rec.currency or "")}
            · Confidence: {escape(rec.confidence)}
            · Evidence: {escape(", ".join(rec.evidence_ids) or "—")}
          </small>
          {('<ul>' + ''.join(f'<li>{escape(a)}</li>' for a in rec.assumptions) + '</ul>') if rec.assumptions else ''}
        </li>
        """
        for rec in analytics.recommendations
    ) or "<li>No recommendations generated.</li>"

    evidence_rows = []
    for field in extraction.fields.values():
        for evidence in field.evidence:
            evidence_rows.append(
                f"""
                <tr>
                  <td><code>{escape(evidence.id)}</code></td>
                  <td>{escape(field.label)}</td>
                  <td>{escape(str(field.value))}{escape(" " + field.unit if field.unit else "")}</td>
                  <td>p.{evidence.page}{f", line {evidence.line}" if evidence.line else ""}</td>
                  <td>{escape(evidence.source_text)}</td>
                  <td>{evidence.confidence:.0%}</td>
                </tr>
                """
            )

    warnings_html = "".join(f"<li>{escape(w)}</li>" for w in analytics.warnings)
    fields_html = "".join(
        f"""
        <tr>
          <td>{escape(field.label)}</td>
          <td>{escape(str(field.value) if field.value is not None else "—")}{escape(" " + field.unit if field.unit else "")}</td>
          <td>{field.confidence:.0%}</td>
          <td>{'Review' if field.needs_review else 'OK'}</td>
          <td>{escape(", ".join(e.id for e in field.evidence))}</td>
        </tr>
        """
        for field in extraction.fields.values()
    ) or "<tr><td colspan='5'>No fields extracted.</td></tr>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoadIQ bill report — {escape(document.filename)}</title>
<style>
  :root {{ --ink:#111813; --paper:#F2F0E9; --moss:#0F3D2E; --lime:#D8F24D; --clay:#C75B32; --line:rgba(17,24,19,.18); }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.55; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 48px 28px 72px; }}
  h1 {{ font-size: 2.2rem; line-height:1.05; margin:0 0 8px; letter-spacing:-.03em; }}
  h2 {{ font-size: .85rem; text-transform:uppercase; letter-spacing:.14em; margin:36px 0 14px; color:var(--moss); }}
  .muted {{ color:rgba(17,24,19,.55); }}
  .meta {{ display:flex; flex-wrap:wrap; gap:8px 18px; font-size:.8rem; color:rgba(17,24,19,.55); border-bottom:1px solid var(--line); padding-bottom:18px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-top:22px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:14px 16px; }}
  .card span {{ display:block; font-size:.66rem; letter-spacing:.12em; text-transform:uppercase; color:rgba(17,24,19,.45); }}
  .card strong {{ display:block; font-size:1.35rem; margin-top:6px; letter-spacing:-.02em; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:28px; }}
  @media (max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .bar-row {{ display:grid; grid-template-columns:120px 1fr 100px; align-items:center; gap:10px; margin:7px 0; font-size:.78rem; }}
  .bar-track {{ height:9px; border-radius:99px; background:rgba(17,24,19,.08); overflow:hidden; }}
  .bar-fill {{ display:block; height:100%; border-radius:99px; background:var(--moss); }}
  .bar-1 {{ background:var(--clay); }} .bar-2 {{ background:#3A6B8A; }} .bar-3 {{ background:#A5C01D; }}
  .bar-value {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .usage-chart {{ display:flex; align-items:flex-end; gap:8px; min-height:150px; padding:12px; background:#fff; border:1px solid var(--line); border-radius:14px; }}
  .usage-col {{ flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; gap:6px; min-width:0; }}
  .usage-bar {{ width:100%; max-width:34px; background:var(--moss); border-radius:6px 6px 0 0; }}
  .usage-label {{ font-size:.6rem; color:rgba(17,24,19,.5); white-space:nowrap; }}
  ul {{ padding-left:18px; }}
  li.insight, li.recommendation {{ margin:0 0 14px; padding:14px 16px; background:#fff; border:1px solid var(--line); border-radius:12px; list-style:none; }}
  li.insight p, li.recommendation p {{ margin:6px 0; font-size:.88rem; }}
  li.insight small, li.recommendation small {{ color:rgba(17,24,19,.5); font-size:.7rem; }}
  li.warning {{ border-left:3px solid var(--clay); }}
  li.positive {{ border-left:3px solid #5A7D2A; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:.76rem; }}
  th, td {{ text-align:left; padding:9px 11px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ font-size:.62rem; text-transform:uppercase; letter-spacing:.12em; color:rgba(17,24,19,.48); background:rgba(17,24,19,.035); }}
  tr:last-child td {{ border-bottom:0; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.7rem; }}
  .footer {{ margin-top:38px; border-top:1px solid var(--line); padding-top:16px; font-size:.72rem; color:rgba(17,24,19,.5); }}
  @media print {{ body {{ background:#fff; }} main {{ padding:20px; }} .card, table, li.insight, li.recommendation {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<main>
  <h1>Electricity bill intelligence</h1>
  <div class="meta">
    <span>{escape(document.filename)}</span>
    <span>Status: {escape(document.status)}</span>
    <span>Pages: {document.page_count}</span>
    <span>Extracted: {escape(extraction.updated_at)}</span>
  </div>

  <div class="cards">
    <div class="card"><span>Consumption</span><strong>{escape(_fmt(summary.get('consumption_kwh'), " kWh"))}</strong></div>
    <div class="card"><span>Total amount</span><strong>{escape(currency)}{escape(_fmt(summary.get('total_cost')))}</strong></div>
    <div class="card"><span>Billing days</span><strong>{escape(_fmt(summary.get('billing_days')))}</strong></div>
    <div class="card"><span>Avg daily use</span><strong>{escape(_fmt(summary.get('average_daily_kwh'), " kWh"))}</strong></div>
    <div class="card"><span>Effective rate</span><strong>{escape(currency)}{escape(_fmt(summary.get('effective_cost_per_kwh')))}/kWh</strong></div>
    <div class="card"><span>Load factor</span><strong>{escape(f"{summary['load_factor']:.0%}" if summary.get('load_factor') is not None else "—")}</strong></div>
  </div>

  <div class="grid">
    <section>
      <h2>Cost breakdown</h2>
      {_bar_chart(analytics.charts.get('cost_breakdown', []), currency)}
    </section>
    <section>
      <h2>Usage history</h2>
      {_usage_chart(analytics.charts.get('usage_history', []), currency)}
    </section>
  </div>

  <h2>Insights</h2>
  <ul>{insights_html}</ul>

  <h2>Recommendations</h2>
  <ul>{recommendations_html}</ul>

  <h2>Extracted fields</h2>
  <table>
    <thead><tr><th>Field</th><th>Value</th><th>Confidence</th><th>Review</th><th>Evidence</th></tr></thead>
    <tbody>{fields_html}</tbody>
  </table>

  {f"<h2>Validation warnings</h2><ul>{warnings_html}</ul>" if warnings_html else ""}

  <h2>Evidence appendix</h2>
  <table>
    <thead><tr><th>ID</th><th>Field</th><th>Value</th><th>Location</th><th>Source text</th><th>Confidence</th></tr></thead>
    <tbody>{''.join(evidence_rows) or "<tr><td colspan='6'>No evidence captured.</td></tr>"}</tbody>
  </table>

  <div class="footer">
    Generated by LoadIQ. Estimates are based on the extracted values and the
    assumptions listed above; verify against the original bill and your supply
    contract before making decisions. Scanned or unusual layouts may require
    manual review.
  </div>
</main>
</body>
</html>"""
