"""Retrieval over the context notes (TF-IDF vectors + cosine similarity).

Why TF-IDF and not a vector DB / embedding model: there are 10 notes, so a hosted store adds
nothing; TF-IDF is fully deterministic (same input -> bit-identical scores), free, offline,
and costs zero tokens. The retriever only RANKS candidates -- whether a note actually justifies
a cost rise is decided by the verification gates in verdict.py, never by similarity alone.
Swapping in a sentence-embedding model would only change `Retriever.query`.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .notes import Note

COST_VOCAB = "freight cost increase higher costs surcharge diesel fuel toll weather flood festival disruption"


class Retriever:
    def __init__(self, notes: list[Note]):
        self.notes = notes
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._vec.fit_transform(f"{n.applies_to}. {n.text}" for n in notes)

    def query(self, route: str, route_type: str, k: int) -> list[tuple[Note, float]]:
        """Metadata filter first (only notes whose scope covers this route), then rank by similarity."""
        text = f"{route.replace('-', ' ')} {route_type} {COST_VOCAB}"
        scores = cosine_similarity(self._vec.transform([text]), self._matrix)[0]
        scoped = [(n, s) for n, s in zip(self.notes, scores) if n.covers_route(route)]
        scoped.sort(key=lambda p: (-round(p[1], 9), p[0].note_id))
        return [(n, float(s)) for n, s in scoped][:k]
