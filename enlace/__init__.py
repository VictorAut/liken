"""
.. include:: ../README.md
"""

from enlace._strats_library import (
    cosine,
    exact,
    fuzzy,
    jaccard,
    lsh,
    tfidf,
)
from enlace.dedupe import Dedupe


__all__ = [
    "Dedupe",
    "exact",
    "cosine",
    "fuzzy",
    "jaccard",
    "lsh",
    "tfidf",
]
