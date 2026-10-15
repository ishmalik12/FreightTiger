# FreightWatch: a shipping-cost watchdog that doesn't make things up

Flags route-weeks where freight cost per tonne-km jumps out of line, then checks the context notes
for a *real* reason. If no note genuinely applies, the row stays **unexplained** (`flagged = Yes`).

```
pip install -r requirements.txt
python run.py                      # -> outputs/flagged_routes.csv  (the graded file)
python eval/run_eval.py            # eval harness (exit code 1 on failure)
./repro_check.sh                   # 3 runs, byte-identical check
python -m freightwatch.ask "why did Chennai-Bangalore get pricier in March?"   # stretch goal
```

## How it works

| Step | File | What happens |
|---|---|---|
| 1 | `metrics.py` | Validate input, assign each shipment to its Mon-Sun `week_of`, group by `route` (origin-destination) and the given `route_type`. Weekly cost/tonne-km = `sum(cost) / sum(tonnes x km)` (volume-weighted, per the brief's formula). |
| 2 | `metrics.py` | **vs own history**: mean of the route's previous 8 calendar weeks, strictly before the current week. **vs similar routes**: mean cost/tonne-km that week across the *other* routes with the same `route_type`. Both exactly as defined in the brief. |
| 3 | `metrics.py` | **Flag rule** (all in `config.py`): vs-own >= +8%, vs-similar >= 0%, and >= 4 prior weeks. |
| 4 | `notes.py` | Each note is parsed into scope, effective dates, direction (does it say costs went *up*?) and any stated % range. |
| 5 | `retrieval.py` | For each flagged row: metadata-filter notes to this route (or "All Routes"), rank by TF-IDF cosine similarity. |
| 6 | `verdict.py` | A note justifies a rise only if it passes **every** gate: scope, direction, dates overlap the week, magnitude. First one that passes is cited; otherwise **unexplained**. The reason text is filled from the note's own words and dates. |

Only suspicious weeks appear in the output, as in `sample_output_format.csv`. `outputs/weekly_metrics_all.csv` has the numbers for every route-week.

## Headline result (728 route-weeks, 7 routes, 104 weeks each)

11 flagged: **4 justified** (Ahmedabad festival surcharge N002; Chennai-Bangalore flood weeks of Feb 24 and Mar 3, N001; Mumbai-Delhi diesel week, N003) and **7 unexplained** (Delhi-Jaipur Nov 2024 x2; Chennai-Bangalore Mar 10 and Mar 17 2025; Mumbai-Pune Sep 15 and Nov 17 2025; Delhi-Chennai Feb 5 2024).

Worth noticing: Chennai-Bangalore stays ~22% high in the week of Mar 10, **after** N001's own window ended (Mar 8; roads passable Mar 9). Citing the flood note there would be a plausible-sounding fabrication, so it's left unexplained.

## How I decided it's right (`eval/run_eval.py`, output in `outputs/eval_report.txt`)

- **A. Note direction**: 10/10 given notes and 10/10 adversarial paraphrases ("prices fell", "did not cause any surcharge", "costs were unaffected"...).
- **B. Labeled verdicts**: 11/11 real flagged weeks and 13/13 synthetic traps: wrong route, week before/after a window, the "costs not affected" notes (N005, N007, N008), a toll note for routes outside the dataset (N004), and a rise far larger than the note's stated 5-7%.
- **C. Audit of the real output**: every justified row re-checked against the raw notes (route, window overlap, direction, id + date in text).
- **D. Independent maths**: all 728 route-weeks recomputed with plain-Python loops (no pandas, different code path); max difference 0.00005 (rounding).
- **E. Sensitivity**: with seeded spike injection, +15% is caught 99.5%, +30% 100%, +10% 67%, +8% 43%, +5% 15%; 0 false alarms on 245 clean 2024 route-weeks. So the rule trades sensitivity to small rises for trust.
- Also: the three rows in your sample file reproduce exactly (same numbers and verdict types), which confirms the baseline definitions.

**Honest limits of this evidence.** I wrote the labels and the code, and some labels for the real rows were written after seeing what was flagged, so B is a regression/consistency check, not independent ground truth. The thresholds were chosen with this data in view. D and E are the independent parts. Hand-spot-check the 4 justified rows, they're the ones that matter.

## Judgement calls (please challenge these)

1. **Peer gate lowered to 0%.** At +5%, genuine spikes on Delhi-Jaipur were missed (recall at +15% was 87%) because peers differ structurally in price (Delhi-Jaipur is ~9% below Mumbai-Pune, its only Short peer). Detection is driven by own-history; vs-peer is reported and used as a sanity gate. Short and Long have just one peer each, so those peer averages are single-route comparisons.
2. **N003 (diesel, "starting this week") has no end date.** I assume its effect explains rises for **8 weeks** (to 2025-06-29), the length of the trailing baseline, after which the baseline has absorbed the step. This is why Mumbai-Pune Sep/Nov 2025 stay unexplained. It's an assumption, not something the note says.
3. **Magnitude tolerance = 5 pp.** The Mumbai-Delhi 2025-05-12 week is +11.6% vs a stated 5-7%; it passes (limit 12%) so is marked justified, but it's the thinnest justified row. At 4 pp it would flip to unexplained.
4. **Fewer than 8 prior weeks**: the average uses all prior weeks available (no padding), as the brief asks. I don't *flag* before 4 prior weeks (affects no row here).
5. **Delhi-Chennai 2024-02-05 (+8.5%)** is probably ordinary noise that crosses the threshold; expect roughly one such weekly false alarm per ~300 clean route-weeks.
6. **Your sample file's 3rd row is malformed CSV** (the reason has an unquoted comma -> 9 fields). My output quotes it properly; header, column order and formats match exactly.

## Known gap: slow creep

The 8-week trailing average follows a gradual drift, so it can't see it week to week. Mumbai-Pune is ~23% higher in Q4 2025 than in 2024, versus ~6% for most other routes (in line with the diesel note), yet only two of its weeks are flagged. That is exactly the "quietly creeps up" case in the brief. Next step: a second detector comparing a recent 13-week average to a fixed longer baseline, reported separately so the graded baselines stay as specified.

## AI use, reproducibility, cost

- **Retrieval** is TF-IDF + cosine over the notes, with a metadata filter on route scope. I chose it over a vector DB / embedding model because 10 short notes don't need one, it is bit-for-bit deterministic, free and offline. Retrieval only *ranks*; the verification gates decide. Swapping in sentence embeddings would only change `Retriever.query`. I have **not** tested an embedding backend. Note-direction parsing is a rule/cue list tuned to this note style and is brittle on unfamiliar phrasing; a production version would use an LLM at temperature 0 for that one step, still behind the same gates.
- **Generative model: none.** Reasons are templates filled with the note's own sentence, id, dates and the observed numbers, so nothing can be invented. The trade-off is less fluent prose.
- **Reproducibility**: no randomness anywhere in the graded path. `repro_check.sh` ran 3 times with different `PYTHONHASHSEED`s: sha256 of all outputs identical (`outputs/reproducibility_log.txt`). The sensitivity test uses a fixed seed.
- **Cost log** (`outputs/cost_log.json`): LLM calls **0**, input tokens **0**, output tokens **0**, cost **$0**. The full run takes a few seconds on a laptop.
- **Q&A** (`freightwatch/ask.py`) parses route/month/year from the question and answers only from `flagged_routes.csv`. If nothing was flagged it says so instead of guessing.
