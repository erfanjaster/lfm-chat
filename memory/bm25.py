"""
bm25.py — A tiny, dependency-free BM25 search engine for long-term memory.

Engineered for low resource use:
  * Pure Python — no numpy/scipy/sklearn, no embedding model, ~0 extra RAM.
  * In-memory inverted index rebuilt from SQLite on startup (a few ms for
    thousands of memories).
  * Hybrid scoring: BM25 relevance + importance + recency, so memories that
    the assistant keeps using naturally rise to the top ("reinforcement").
  * Lightweight English stemming + stopword removal for better recall
    ("python coding" also matches "Python code", "codes", ...).
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

STOPWORDS = frozenset("""
a an the and or but if then else when while of at by for with about against
between into through during before after above below to from up down in out
on off over under again further once here there all any both each few more
most other some such no nor not only own same so than too very can will just
should now is are was were be been being am do does did doing have has had
having i you he she it we they them his her its our your their me him us
what which who whom this that these those as
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Common English suffixes, longest first (very light "stemmer")
_SUFFIXES = ("ization", "ations", "ation", "ities", "iness", "ments", "ing",
             "ers", "est", "ies", "ied", "ely", "ive", "ful", "ous", "ity",
             "ment", "ness", "able", "ible", "ally", "ed", "es", "er", "ly", "s")


def _stem(word: str) -> str:
    """Strip at most two suffix passes — enough for recall, cheap to run."""
    if len(word) <= 3:
        return word
    for _ in range(2):
        for suf in _SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                word = word[: -len(suf)]
                break
        else:
            break
    # normalize silent trailing 'e'  →  code/codes/coding all become "cod"
    if len(word) >= 4 and word.endswith("e") and not word.endswith(("ee", "ie")):
        word = word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """Lowercase, split, drop stopwords, stem. Keeps code identifiers intact."""
    tokens = _TOKEN_RE.findall(text.lower())
    out: List[str] = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        out.append(_stem(t) if t.isalpha() and len(t) > 3 else t)
    return out


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """Minimal view of a memory row used by the index."""
    mid: int
    text: str
    mtype: str = ""
    importance: float = 0.5
    hits: int = 0
    last_used: float = 0.0
    pinned: bool = False
    tags: str = ""


@dataclass
class _Posting:
    tf: int = 0
    fields: List[str] = field(default_factory=list)


class BM25Index:
    """Okapi BM25 with field weighting (tags count double)."""

    K1 = 1.4
    B = 0.75

    def __init__(self) -> None:
        self._docs: Dict[int, MemoryRecord] = {}
        self._tf: Dict[int, Counter] = {}          # mid -> term freq
        self._df: Counter = Counter()              # term -> doc freq
        self._len: Dict[int, int] = {}
        self._avg_len: float = 1.0

    # -- index maintenance -------------------------------------------------

    def add(self, rec: MemoryRecord) -> None:
        self.remove(rec.mid)
        toks = tokenize(rec.text)
        # tags are searched with double weight
        toks += tokenize(rec.tags) * 2
        tf = Counter(toks)
        self._docs[rec.mid] = rec
        self._tf[rec.mid] = tf
        self._len[rec.mid] = len(toks) or 1
        for t in tf:
            self._df[t] += 1
        self._recompute_avg()

    def remove(self, mid: int) -> None:
        if mid in self._docs:
            for t in self._tf[mid]:
                self._df[t] -= 1
                if self._df[t] <= 0:
                    del self._df[t]
            del self._docs[mid]
            del self._tf[mid]
            del self._len[mid]
            self._recompute_avg()

    def update(self, rec: MemoryRecord) -> None:
        self.add(rec)

    def clear(self) -> None:
        self._docs.clear(); self._tf.clear(); self._df.clear(); self._len.clear()

    def _recompute_avg(self) -> None:
        n = len(self._len)
        self._avg_len = (sum(self._len.values()) / n) if n else 1.0

    def __len__(self) -> int:
        return len(self._docs)

    # -- scoring -------------------------------------------------------------

    def _bm25_scores(self, query_tokens: List[str]) -> Dict[int, float]:
        n_docs = len(self._docs)
        if n_docs == 0:
            return {}
        scores: Dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for mid in self._postings_for(term):
                tf = self._tf[mid][term]
                dl = self._len[mid]
                denom = tf + self.K1 * (1 - self.B + self.B * dl / self._avg_len)
                scores[mid] += idf * (tf * (self.K1 + 1)) / denom
        return dict(scores)

    def _postings_for(self, term: str) -> Iterable[int]:
        # Linear scan is fine up to ~10k memories (single-digit ms);
        # keeps memory tiny and code simple.
        for mid, tf in self._tf.items():
            if term in tf:
                yield mid

    @staticmethod
    def _recency_boost(rec: MemoryRecord, now: float) -> float:
        age_days = max(0.0, (now - (rec.last_used or 0)) / 86400.0)
        return math.exp(-age_days / 30.0)  # half-life ~3 weeks

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.35,
               type_boost: Dict[str, float] | None = None) -> List[Tuple[MemoryRecord, float]]:
        """Hybrid search: 0.72*BM25 + 0.16*importance + 0.12*recency (+boosts)."""
        q_toks = tokenize(query)
        if not q_toks or not self._docs:
            return []
        raw = self._bm25_scores(q_toks)
        if not raw:
            return []
        max_raw = max(raw.values()) or 1.0
        now = time.time()
        type_boost = type_boost or {}
        results: List[Tuple[MemoryRecord, float]] = []
        for mid, r in raw.items():
            rec = self._docs[mid]
            bm25_n = r / max_raw                     # 0..1
            imp = max(0.0, min(1.0, rec.importance))
            rec_b = self._recency_boost(rec, now)
            score = 0.72 * bm25_n + 0.16 * imp + 0.12 * rec_b
            score += 0.20 * type_boost.get(rec.mtype, 0.0)
            # tiny bonus for memories that proved useful before
            score += 0.02 * min(rec.hits, 10) / 10.0
            results.append((rec, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return [(rec, s) for rec, s in results[:top_k] if s >= min_score]
