"""Requirement 4: deterministic, template-based dispatch notices (no LLM, no network)."""
from datetime import date
from pathlib import Path
from jinja2 import Template


def _out_lines(ward_id, transfers):
    if transfers is None or not len(transfers):
        return []
    t = transfers[transfers.from_ward == ward_id]
    return [f"{int(r.patients)} pts -> {r.to_ward}" for r in t.itertuples()]


def sms(r, week, transfers=None):
    t = (f"DENGUE ALERT {week} {r.ward_id} ({r.city}) Risk:{r.tier} Cases:{r.pred_cases:.0f} "
         f"Saline {r.saline_allocated}/{r.saline_bags_needed} Kits {r.kits_allocated}/"
         f"{r.diagnostic_kits_needed} Beds {r.beds_allocated}/{r.hospital_beds_required} "
         f"Crews {r.crews_assigned}/{r.crews_needed}")
    if r.alert != "OK":
        t += f" DEFICIT:{r.deficit_items}"
    out = _out_lines(r.ward_id, transfers)
    if out:
        t += " ROUTE:" + "; ".join(out)
    return t


def whatsapp(r, week, transfers=None):
    s = f"*Dengue Dispatch: {r.ward_id}, {r.city}*\n_Forecast from {week}, window +2 to +4 weeks_\n"
    if r.alert != "OK":
        s += f"\U0001F534 *{r.alert}* ({r.deficit_items})\n"
    s += (f"Risk: *{r.tier}*\nPredicted cases: {r.pred_cases:.0f}\n"
          f"Saline: {r.saline_allocated} allocated / {r.saline_bags_needed} required\n"
          f"Test kits: {r.kits_allocated} / {r.diagnostic_kits_needed}\n"
          f"Beds: {r.beds_allocated} / {r.hospital_beds_required}\n"
          f"Spray teams assigned: {r.crews_assigned} (required {r.crews_needed})")
    out = _out_lines(r.ward_id, transfers)
    if out:
        s += "\nOverflow routing: " + "; ".join(out)
    return s


BRIEF = Template("""<!doctype html><html><head><meta charset="utf-8"><title>Brief {{ city }}</title>
<style>body{font-family:Arial,sans-serif;margin:28px}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border:1px solid #999;padding:5px;text-align:left}th{background:#eee}.bad{color:#b00020;font-weight:bold}
.Critical{background:#f8d7da}.High{background:#ffe5d0}@media print{button{display:none}}</style></head><body>
<h1>Dengue Commissioner Brief: {{ city }}</h1>
<p>Forecast from {{ week }} (window +2 to +4 weeks) | Generated {{ today }} | Offline template, no external services</p>
<button onclick="window.print()">Print / Save as PDF</button>
<table><tr><th>Ward</th><th>Tier</th><th>Pred. cases</th><th>Saline</th><th>Kits</th><th>Beds</th>
<th>Spray teams</th><th>Alert</th><th>Overflow routing</th></tr>
{% for r in rows %}<tr class="{{ r.tier }}"><td>{{ r.ward_id }}</td><td>{{ r.tier }}</td><td>{{ '%.0f'|format(r.pred_cases) }}</td>
<td>{{ r.saline_allocated }}/{{ r.saline_bags_needed }}</td><td>{{ r.kits_allocated }}/{{ r.diagnostic_kits_needed }}</td>
<td>{{ r.beds_allocated }}/{{ r.hospital_beds_required }}</td><td>{{ r.crews_assigned }}/{{ r.crews_needed }}</td>
<td class="{{ 'bad' if r.alert!='OK' else '' }}">{{ r.alert }}{% if r.deficit_items %}<br>{{ r.deficit_items }}{% endif %}</td>
<td>{{ route[r.ward_id] }}</td></tr>{% endfor %}</table>
<p>Format: allocated / required. Tiers are percentile-based model outputs.</p></body></html>""")


def brief_html(alloc, city, week, transfers=None):
    rows = list(alloc[alloc.city == city].itertuples())
    route = {r.ward_id: "; ".join(_out_lines(r.ward_id, transfers)) for r in rows}
    return BRIEF.render(rows=rows, city=city, week=week, today=date.today(), route=route)


def export_all(alloc, week, transfers=None, out="exports", n=10):
    p = Path(out)
    p.mkdir(exist_ok=True)
    top = alloc.head(n)
    (p / "sms_samples.txt").write_text("\n".join(sms(r, week, transfers) for r in top.itertuples()), encoding="utf-8")
    (p / "whatsapp_samples.txt").write_text(
        "\n\n------\n\n".join(whatsapp(r, week, transfers) for r in top.itertuples()), encoding="utf-8")
    for c in alloc.city.unique():
        (p / f"brief_{c.replace(' ', '_')}.html").write_text(brief_html(alloc, c, week, transfers), encoding="utf-8")