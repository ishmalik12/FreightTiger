"""Stretch goal: plain-English questions answered ONLY from the pipeline's own results.

    python -m freightwatch.ask "why did Chennai-Bangalore get pricier in March 2025?"

It parses route / month / year from the question, looks them up in outputs/flagged_routes.csv,
and answers with the stored numbers, verdict and cited note. If nothing was flagged it says so and
quotes the weekly figures instead of inventing a cause. No LLM, so answers are reproducible.
"""
import re
import sys
from pathlib import Path

import pandas as pd

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def parse_question(q: str, routes: list[str]):
    ql = q.lower()
    hits = [r for r in routes if all(city.lower() in ql for city in r.split("-"))]
    if not hits:  # fall back to routes where at least one city is named
        hits = [r for r in routes if any(city.lower() in ql for city in r.split("-"))]
    month = next((i for i, m in enumerate(MONTHS, 1) if re.search(rf"\b{m}\b|\b{m[:3]}\b", ql)), None)
    y = re.search(r"\b(20\d\d)\b", ql)
    return hits, month, int(y[1]) if y else None


def answer(q: str, out_dir="outputs") -> str:
    out_dir = Path(out_dir)
    flagged = pd.read_csv(out_dir / "flagged_routes.csv", parse_dates=["week_of"]).fillna("")
    weekly = pd.read_csv(out_dir / "weekly_metrics_all.csv", parse_dates=["week_of"])
    routes, month, year = parse_question(q, sorted(weekly.route.unique()))
    if not routes:
        return "I couldn't tell which route you mean. Known routes: " + ", ".join(sorted(weekly.route.unique())) + "."
    lines = []
    for route in routes:
        f, w = flagged[flagged.route == route], weekly[weekly.route == route]
        if month:
            f, w = f[f.week_of.dt.month == month], w[w.week_of.dt.month == month]
        if year:
            f, w = f[f.week_of.dt.year == year], w[w.week_of.dt.year == year]
        scope = f"{route}" + (f" in {MONTHS[month - 1].title()}" if month else "") + (f" {year}" if year else "")
        if f.empty:
            if w.empty:
                lines.append(f"{scope}: no shipment data for that period.")
            else:
                lines.append(f"{scope}: no week was flagged. Cost per tonne-km ranged "
                             f"{w.cost_per_tonne_km.min():.2f} to {w.cost_per_tonne_km.max():.2f} across {len(w)} weeks, "
                             f"so there is no unusual rise to explain.")
            continue
        lines.append(f"{scope}: {len(f)} week(s) flagged as unusually expensive.")
        for r in f.itertuples():
            verdict = "justified" if r.flagged.startswith("No") else "UNEXPLAINED"
            lines.append(f"  - Week of {r.week_of.date()}: {r.cost_per_tonne_km:.2f} per tonne-km, {r.vs_own_history}, "
                         f"{r.vs_similar_routes} [{verdict}"
                         f"{', note ' + r.matched_note_id if r.matched_note_id else ''}]\n    {r.reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:]) or input("Question: ")))
