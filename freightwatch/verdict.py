"""Decides 'justified' vs 'unexplained' for a flagged route-week, and writes the reason.

A note may justify a rise ONLY if it passes every gate, in this order:
  1. scope      - applies to this route (or says 'All Routes')
  2. direction  - it says costs went UP (a note saying 'not affected' never counts)
  3. window     - its effective dates overlap the Monday-Sunday week
  4. magnitude  - if it states a % range, the observed rise isn't far beyond it
No note passes -> 'unexplained'. Reasons are filled from the note's own text and dates (no free generation).
"""
from dataclasses import dataclass

import pandas as pd

from .config import Config
from .notes import Note
from .retrieval import Retriever

REJECT_WINDOW, REJECT_DIRECTION, REJECT_MAGNITUDE = "window", "direction", "magnitude"


@dataclass(frozen=True)
class Verdict:
    justified: bool
    note_id: str
    reason: str
    retrieved: tuple


def _gap_days(week_start: pd.Timestamp, note: Note) -> int:
    week_end = week_start + pd.Timedelta(days=6)
    if note.window_end < week_start:
        return (week_start - note.window_end).days
    if note.window_start > week_end:
        return (note.window_start - week_end).days
    return 0


def _check(note: Note, week: pd.Timestamp, vs_own: float, cfg: Config) -> str | None:
    """Return None if the note justifies the rise, else the first gate it failed."""
    if not note.is_cost_driver:          # a note that never says "costs rose" can't justify a rise, any dates
        return REJECT_DIRECTION
    if _gap_days(week, note) > 0:
        return REJECT_WINDOW
    if note.pct_range and vs_own > note.pct_range[1] + cfg.magnitude_tolerance_pp:
        return REJECT_MAGNITUDE
    return None


def judge(row, retriever: Retriever, cfg: Config = Config()) -> Verdict:
    week, vs_own = row.week_of, row.vs_own_pct
    hits = retriever.query(row.route, row.route_type, cfg.retrieval_top_k)  # gate 1 (scope) is the filter
    retrieved = tuple(n.note_id for n, _ in hits)

    rejected = []
    for note, _ in hits:
        failed = _check(note, week, vs_own, cfg)
        if failed is None:
            return Verdict(True, note.note_id, _justified_reason(note, vs_own), retrieved)
        rejected.append((note, failed))

    # Unexplained. Mention a note only if it is genuinely close in time. Preference order:
    # a note overlapping the week, then notes that WOULD have explained a rise (cost drivers), then the rest.
    near = [((g > 0, not n.is_cost_driver, g, i), n, f) for i, (n, f) in enumerate(rejected)
            if (g := _gap_days(week, n)) <= cfg.near_note_days]
    if not near:
        return Verdict(False, "", "No matching note found for this route or date range. "
                       "Cost rise looks unexplained and worth a human review.", retrieved)
    _, n, failed = min(near, key=lambda t: t[0])
    return Verdict(False, "", _rejected_reason(n, failed, week, vs_own, cfg), retrieved)


def _justified_reason(note: Note, vs_own: float) -> str:
    s = (f'Matches note {note.note_id} dated {note.date.date()} '
         f'(applies to {note.applies_to}; effective {note.window_start.date()} to {note.window_end.date()}): '
         f'"{note.first_sentence()}."')
    if note.pct_range:
        lo, hi = note.pct_range
        s += (f" The note puts the effect at roughly {lo:g}-{hi:g}%; the observed {vs_own:+.1f}% vs own history "
              f"is within normal weekly noise of that range.")
    if "open-ended" in note.window_basis:
        s += f" ({note.window_basis}.)"
    return s + " The cost rise has a clear explanation."


def _rejected_reason(note: Note, failed: str, week: pd.Timestamp, vs_own: float, cfg: Config) -> str:
    head = f"The closest note ({note.note_id}, {note.date.date()})"
    if failed == REJECT_WINDOW:
        body = (f'concerns {note.applies_to} ("{note.first_sentence()}") but its effective window '
                f"({note.window_start.date()} to {note.window_end.date()}) does not cover the week of {week.date()}")
    elif failed == REJECT_DIRECTION:
        body = f'reads "{note.first_sentence()}" -- it does not describe a reason for a cost rise on this route'
    else:
        body = (f'covers this week but states only ~{note.pct_range[0]:g}-{note.pct_range[1]:g}%, far short of the '
                f"observed {vs_own:+.1f}%")
    return f"{head} {body}. No genuine justification found; flagged for review."
