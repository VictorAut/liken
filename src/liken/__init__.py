from liken._dedupers import cosine
from liken._dedupers import exact
from liken._dedupers import fuzzy
from liken._dedupers import jaccard
from liken._dedupers import lsh
from liken._dedupers import tfidf
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
