"""
.. include:: ../README.md
"""

from enlace._strats_library import cosine
from enlace._strats_library import exact
from enlace._strats_library import fuzzy
from enlace._strats_library import jaccard
from enlace._strats_library import lsh
from enlace._strats_library import tfidf
from enlace.dedupe import Dedupe


__all__ = [
    "Dedupe",
    "exact",
    "fuzzy",
    "lsh",
    "tfidf",
    "cosine",
    "jaccard",
]
