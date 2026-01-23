"""
.. include:: ../README.md
"""

from dupegrouper.dedupe import Duped
from dupegrouper._strats_library import (
    cosine,
    exact,
    fuzzy,
    jaccard,
    lsh,
    tfidf,
)

__all__ = [
    "Duped",
    "exact",
    "cosine",
    "fuzzy",
    "jaccard",
    "lsh",
    "tfidf",
]
