"""Turns a free-text context note into structured facts we can verify against:
scope (which routes), time window, whether it describes a cost *increase*, and any stated magnitude.

This is deliberately rule-based: it is deterministic, free, and every decision can be audited.
Trade-off: the cue lists are tuned to this note style and are brittle on unseen phrasings
(see eval/ for the adversarial cases and README for the production alternative).
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .config import Config

ALL_ROUTES = "all routes"

# Phrases that say costs went UP.
DRIVER_CUES = [
    r"surcharge", r"pushing up", r"higher (?:trip )?costs?",
    r"costs? (?:rose|rise|increased?|went up)", r"(?:prices?|rates?) (?:rose|rise|increased?|hikes?)",
]
# Phrases that say costs were NOT affected / conditions normal. These veto a driver reading.
NEUTRAL_CUES = [
    r"not significantly affected", r"costs? (?:were|was|remained) (?:not|unchanged|unaffected)",
    r"no (?:significant|major|material) (?:disruptions?|impact)", r"remained (?:stable|normal)",
    r"returned to normal", r"without a rate change", r"not part of this dataset",
    r"improved road conditions", r"unaffected", r"no (?:change|impact)",
]
NEGATION = re.compile(r"\b(?:not|no|never|without|didn't|did not|won't|n't)\b[^.]{0,25}$", re.I)
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


@dataclass(frozen=True)
class Note:
    note_id: str
    date: pd.Timestamp
    applies_to: str
    text: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    window_basis: str
    is_cost_driver: bool
    polarity_basis: str
    pct_range: tuple | None   # (low, high) if the note states a magnitude

    @property
    def all_routes(self) -> bool:
        return self.applies_to.strip().lower() == ALL_ROUTES

    def covers_route(self, route: str) -> bool:
        return self.all_routes or self.applies_to.strip().lower() == route.lower()

    def first_sentence(self) -> str:
        return re.split(r"(?<=\.)\s", self.text.strip())[0].rstrip(".")


def classify_polarity(text: str) -> tuple[bool, str]:
    t = text.lower()
    for pat in NEUTRAL_CUES:
        if re.search(pat, t):
            return False, f"neutral cue /{pat}/"
    for pat in DRIVER_CUES:
        for m in re.finditer(pat, t):
            if not NEGATION.search(t[:m.start()][-40:]):
                return True, f"driver cue /{pat}/"
    return False, "no cost-increase cue"


def parse_window(text: str, d: pd.Timestamp, cfg: Config) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    m = re.search(r"from ([A-Za-z]{3})[a-z]* (\d{1,2}) (?:to|through|until) ([A-Za-z]{3})[a-z]* (\d{1,2})", text)
    if m and m[1].lower() in MONTHS and m[3].lower() in MONTHS:
        start = pd.Timestamp(date(d.year, MONTHS[m[1].lower()], int(m[2])))
        end = pd.Timestamp(date(d.year, MONTHS[m[3].lower()], int(m[4])))
        if end < start:
            end = end.replace(year=end.year + 1)
        return start, end, "explicit dates in the note"
    if re.search(r"\bstarting\b|\bfrom this week\b|\bbegan\b", text.lower()):
        end = d + timedelta(weeks=cfg.open_ended_weeks) - timedelta(days=1)
        return d, end, f"open-ended note: effect assumed for {cfg.open_ended_weeks} weeks from its date"
    return d, d + timedelta(days=6), "single week from the note date"


def parse_pct_range(text: str) -> tuple | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m[1]), float(m[2])
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return (float(m[1]), float(m[1])) if m else None


def parse_note(row, cfg: Config = Config()) -> Note:
    d = pd.Timestamp(row["date"])
    start, end, basis = parse_window(row["note"], d, cfg)
    driver, pol = classify_polarity(row["note"])
    return Note(row["note_id"], d, str(row["applies_to"]).strip(), row["note"].strip(),
                start, end, basis, driver, pol, parse_pct_range(row["note"]) if driver else None)


def load_notes(path, cfg: Config = Config()) -> list[Note]:
    df = pd.read_csv(path)
    return [parse_note(r, cfg) for r in df.to_dict("records")]
