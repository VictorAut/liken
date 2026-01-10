"""
.. include:: ../README.md
"""

from dupegrouper.base import Duped
from dupegrouper.custom import register
from dupegrouper.strats_library import (
    exact,
    cosine,
    fuzzy,
    jaccard,
    lsh,
    tfidf,
    str_contains,
    str_endswith,
    str_startswith,
)


__all__ = [
    "Duped",
    "exact",
    "cosine",
    "fuzzy",
    "jaccard",
    "lsh",
    "tfidf",
    "str_contains",
    "str_endswith",
    "str_startswith",
    "register",
]
