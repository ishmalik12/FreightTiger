"""Every tunable number lives here, so the README can point at one place."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Comparison baselines (fixed by the brief -- do not change for graded output).
    history_weeks: int = 8

    # Flag rule: a route-week is a candidate when ALL of these hold.
    min_vs_own_pct: float = 8.0     # rise vs trailing 8-week average (weekly noise is ~3-4%)
    min_vs_peer_pct: float = 0.0    # sanity gate only: not cheaper than similar routes. Peers differ in price level
                                    # (Delhi-Jaipur sits ~9% below Mumbai-Pune), so a high bar made blind spots.
    min_prior_weeks: int = 4        # need a few weeks of history before trusting the baseline

    # Note applicability.
    open_ended_weeks: int = 8       # "starting this week" notes: effect assumed for this long.
                                    # After that the trailing baseline has absorbed the step.
    magnitude_tolerance_pp: float = 5.0   # allowed overshoot above a note's stated % range
    near_note_days: int = 28        # how close a non-matching note must be to be mentioned
    retrieval_top_k: int = 5
