"""Eval harness. Run from the project root:  python eval/run_eval.py   (exit code 1 on any failure)

 A. note polarity     - hand-labeled: the 10 given notes + adversarial paraphrases
 B. verdict labels    - hand-labeled real flags + synthetic 'trap' rows (wrong route / window / 'no effect' note / too big)
 C. output audit      - every justified row in the real output re-checked against the raw notes
 D. independent maths - weekly numbers recomputed with plain-Python loops, no pandas, compared to the pipeline
 E. sensitivity       - inject known spikes (seeded) and measure detection; measure false alarms on clean weeks
"""
import csv
import random
import sys
from collections import defaultdict, namedtuple
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from freightwatch.config import Config                                    # noqa: E402
from freightwatch.metrics import add_baselines, is_candidate              # noqa: E402
from freightwatch.notes import classify_polarity, load_notes              # noqa: E402
from freightwatch.pipeline import build_weekly, run                       # noqa: E402
from freightwatch.retrieval import Retriever                              # noqa: E402
from freightwatch.verdict import judge                                    # noqa: E402

CFG = Config()
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


# ---------- A. note polarity -------------------------------------------------------------------------------
GIVEN = {"N001": True, "N002": True, "N003": True, "N004": False, "N005": False,
         "N006": False, "N007": False, "N008": False, "N009": False, "N010": False}
ADVERSARIAL = [
    ("Diesel prices fell by 3% this week, lowering costs.", False),
    ("Toll rates on the Mumbai-Pune expressway were raised, pushing up costs by 4%.", True),
    ("Heavy rain caused delays but did not cause any surcharge.", False),
    ("Transporters levied a festival surcharge of 8-10% on this corridor.", True),
    ("Costs were unaffected by the festival season.", False),
    ("Monsoon disruption led to higher trip costs for a fortnight.", True),
    ("No change in rates was observed despite the strike.", False),
    ("Fuel prices rose 6% starting this week.", True),
    ("The new toll plaza is expected to raise tariffs next year.", False),
    ("Freight rates increased sharply due to truck shortage.", True),
]


def eval_polarity():
    print("A. Note polarity (is this note a reason for a cost INCREASE?)")
    notes = {n.note_id: n for n in load_notes(ROOT / "data/context_notes.csv", CFG)}
    got = {k: notes[k].is_cost_driver for k in GIVEN}
    check("given notes (10)", got == GIVEN, f"{sum(got[k] == GIVEN[k] for k in GIVEN)}/10")
    bad = [(t, e) for t, e in ADVERSARIAL if classify_polarity(t)[0] != e]
    check("adversarial paraphrases", not bad, f"{len(ADVERSARIAL) - len(bad)}/{len(ADVERSARIAL)}")
    for t, e in bad:
        print(f"      missed (expected driver={e}): {t}")
    return notes


# ---------- B. labeled verdicts ----------------------------------------------------------------------------
def eval_verdicts(weekly, notes):
    print("B. Hand-labeled verdicts")
    retr = Retriever(list(notes.values()))
    Row = namedtuple("Row", "route route_type week_of vs_own_pct")
    labels = pd.read_csv(ROOT / "eval/labeled_cases.csv", dtype=str, keep_default_na=False)
    wk = weekly.set_index(["route", "week_of"])
    stats = defaultdict(lambda: [0, 0])
    for c in labels.itertuples():
        week = pd.Timestamp(c.week_of)
        vs = float(c.vs_own_pct) if c.vs_own_pct else wk.loc[(c.route, week), "vs_own_pct"]
        v = judge(Row(c.route, c.route_type, week, vs), retr, CFG)
        got_flag = "No (justified)" if v.justified else "Yes"
        ok = got_flag == c.expected_flagged and v.note_id == c.expected_note
        stats[c.source][0] += ok
        stats[c.source][1] += 1
        if not ok:
            print(f"      MISS {c.route} {c.week_of}: got {got_flag}/{v.note_id!r}, want {c.expected_flagged}/{c.expected_note!r} ({c.why})")
    for src, (ok, n) in stats.items():
        check(f"{src} cases", ok == n, f"{ok}/{n}")
    # Detection side of the real labels: the pipeline's candidate set must equal the labeled real set.
    cand = set(map(tuple, weekly[is_candidate(weekly, CFG)][["route", "week_of"]].assign(
        week_of=lambda d: d.week_of.dt.strftime("%Y-%m-%d")).values))
    want = {(r.route, r.week_of) for r in labels[labels.source == "real"].itertuples()}
    check("candidate weeks == labeled real set", cand == want, f"(extra={sorted(cand - want)}, missing={sorted(want - cand)})")


# ---------- C. audit of the real output --------------------------------------------------------------------
def eval_audit(notes):
    print("C. Audit of outputs/flagged_routes.csv against raw notes")
    out = pd.read_csv(ROOT / "outputs/flagged_routes.csv", dtype=str, keep_default_na=False)
    problems = []
    for r in out.itertuples():
        wk = pd.Timestamp(r.week_of)
        if r.flagged == "Yes":
            if r.matched_note_id:
                problems.append(f"{r.route} {r.week_of}: unexplained row has note id")
            continue
        n = notes.get(r.matched_note_id)
        if n is None:
            problems.append(f"{r.route} {r.week_of}: cites unknown note {r.matched_note_id!r}")
            continue
        if not n.covers_route(r.route):
            problems.append(f"{r.route} {r.week_of}: note {n.note_id} is for {n.applies_to}")
        if not (n.window_start <= wk + pd.Timedelta(days=6) and n.window_end >= wk):
            problems.append(f"{r.route} {r.week_of}: note {n.note_id} window does not overlap the week")
        if not n.is_cost_driver:
            problems.append(f"{r.route} {r.week_of}: note {n.note_id} does not describe a cost rise")
        if n.note_id not in r.reason or str(n.date.date()) not in r.reason:
            problems.append(f"{r.route} {r.week_of}: reason text lacks the note id/date")
    check("justified rows cite an applicable, in-window, cost-rise note", not problems, f"({len(out)} rows)")
    for p in problems:
        print("      " + p)


# ---------- D. independent recomputation -------------------------------------------------------------------
def eval_independent_maths():
    print("D. Independent recomputation (plain Python, no pandas) vs pipeline numbers")
    tot_cost, tot_tkm, rtype = defaultdict(float), defaultdict(float), {}
    with open(ROOT / "data/shipment_records.csv", newline="") as f:
        for r in csv.DictReader(f):
            d = datetime.strptime(r["shipment_date"], "%Y-%m-%d").date()
            monday = d - timedelta(days=d.weekday())
            key = (r["origin"] + "-" + r["destination"], monday)
            tot_cost[key] += float(r["freight_cost_inr"])
            tot_tkm[key] += float(r["quantity_tonnes"]) * float(r["distance_km"])
            rtype[key[0]] = r["route_type"]
    cpt = {k: tot_cost[k] / tot_tkm[k] for k in tot_cost}
    by_route = defaultdict(list)
    for (route, wk) in sorted(cpt):
        by_route[route].append(wk)
    full = pd.read_csv(ROOT / "outputs/weekly_metrics_all.csv")
    worst = 0.0
    n_checked = 0
    for r in full.itertuples():
        wk = date.fromisoformat(r.week_of)
        weeks = by_route[r.route]
        prior = [cpt[(r.route, w)] for w in weeks if wk - timedelta(weeks=8) <= w < wk]
        peers = [cpt[(o, wk)] for o in by_route if o != r.route and rtype[o] == rtype[r.route] and (o, wk) in cpt]
        worst = max(worst, abs(cpt[(r.route, wk)] - r.cost_per_tonne_km))
        if prior:
            worst = max(worst, abs(100 * (cpt[(r.route, wk)] / (sum(prior) / len(prior)) - 1) - r.vs_own_pct) / 100)
        if peers:
            worst = max(worst, abs(100 * (cpt[(r.route, wk)] / (sum(peers) / len(peers)) - 1) - r.vs_peer_pct) / 100)
        n_checked += 1
    check("cost/tonne-km, vs-own, vs-peers agree on every route-week", worst < 1e-3,
          f"({n_checked} rows, max abs diff {worst:.6f} after rounding)")


# ---------- E. sensitivity ---------------------------------------------------------------------------------
def eval_sensitivity(weekly, trials=200, seed=7):
    print("E. Sensitivity: inject a one-week spike of +X% into clean 2024 weeks (seed fixed) and see if it is caught")
    rng = random.Random(seed)
    base = weekly[["route", "route_type", "week_of", "cost_per_tonne_km"]].copy()
    pool = base[(base.week_of >= "2024-03-04") & (base.week_of <= "2024-10-28")]
    pool = pool[~((pool.route == "Delhi-Jaipur") & (pool.week_of >= "2024-11-01"))]
    idx = list(pool.index)
    results = {}
    for pct in (5, 8, 10, 15, 30):
        hits = 0
        for _ in range(trials):
            i = rng.choice(idx)
            t = base.copy()
            t.loc[i, "cost_per_tonne_km"] *= 1 + pct / 100
            w = add_baselines(t, CFG)
            hits += bool(is_candidate(w, CFG).loc[w[(w.route == t.loc[i, "route"]) & (w.week_of == t.loc[i, "week_of"])].index[0]])
        results[pct] = hits / trials
    for pct, rec in results.items():
        print(f"      +{pct:>2}% spike detected in {rec:5.1%} of {trials} trials")
    check("+15% spikes caught >= 95%", results[15] >= 0.95)
    check("+30% spikes caught 100%", results[30] == 1.0)
    clean = weekly[(weekly.week_of >= "2024-03-04") & (weekly.week_of <= "2024-10-28")]
    fp = int(is_candidate(clean, CFG).sum())
    check("false alarms on 2024 Mar-Oct (no known events)", fp <= 2, f"{fp} of {len(clean)} route-weeks ({fp / len(clean):.1%})")


if __name__ == "__main__":
    run(ROOT / "data/shipment_records.csv", ROOT / "data/context_notes.csv", ROOT / "outputs", CFG)
    weekly = build_weekly(ROOT / "data/shipment_records.csv", CFG)
    notes = eval_polarity()
    eval_verdicts(weekly, notes)
    eval_audit(notes)
    eval_independent_maths()
    eval_sensitivity(weekly)
    print("\nRESULT:", "ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
    sys.exit(1 if failures else 0)
