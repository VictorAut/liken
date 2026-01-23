"""
.. include:: ../README.md
"""

from dupegrouper._strats_library import (
    cosine,
    exact,
    fuzzy,
    jaccard,
    lsh,
    tfidf,
)
from dupegrouper.dedupe import Dedupe


__all__ = [
    "Dedupe",
    "exact",
    "cosine",
    "fuzzy",
    "jaccard",
    "lsh",
    "tfidf",
]
