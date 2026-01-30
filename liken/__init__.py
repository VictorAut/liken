from liken._strats_library import cosine
from liken._strats_library import exact
from liken._strats_library import fuzzy
from liken._strats_library import jaccard
from liken._strats_library import lsh
from liken._strats_library import tfidf
from liken.dedupe import Dedupe


__all__ = [
    "Dedupe",
    "exact",
    "fuzzy",
    "lsh",
    "tfidf",
    "cosine",
    "jaccard",
]
