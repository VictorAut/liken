import importlib

from liken import custom
from liken import datasets
from liken import preprocessors
from liken.backends.pandas.affordances import register_pd_affordances
from liken.collections.pipelines import Col
from liken.collections.pipelines import Pipeline
from liken.collections.pipelines import col
from liken.collections.pipelines import pipeline
from liken.dedupers.cosine import cosine
from liken.dedupers.exact import exact
from liken.dedupers.fuzzy import fuzzy
from liken.dedupers.isin import isin
from liken.dedupers.isna import isna
from liken.dedupers.jaccard import jaccard
from liken.dedupers.lsh import lsh
from liken.dedupers.str_contains import str_contains
from liken.dedupers.str_endswith import str_endswith
from liken.dedupers.str_len import str_len
from liken.dedupers.str_startswith import str_startswith
from liken.dedupers.tfidf import tfidf
from liken.liken import Dedupe
from liken.liken import dedupe


__all__ = [
    "dedupe",
    "Dedupe",
    "exact",
    "fuzzy",
    "lsh",
    "tfidf",
    "cosine",
    "jaccard",
    "isin",
    "isna",
    "str_contains",
    "str_endswith",
    "str_len",
    "str_startswith",
    "pipeline",
    "Pipeline",
    "col",
    "Col",
    "custom",
    "preprocessors",
    "datasets",
]


# registers pandas affordances
register_pd_affordances()


# "force" upfront backend registration
importlib.import_module("liken.backends.pandas.backend")
importlib.import_module("liken.backends.polars.backend")
importlib.import_module("liken.backends.modin.backend")
importlib.import_module("liken.backends.pyspark.backend")
importlib.import_module("liken.backends.dask.backend")
importlib.import_module("liken.backends.ray.backend")
