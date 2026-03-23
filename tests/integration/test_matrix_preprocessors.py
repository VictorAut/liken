"""Narrow integration tests for creation of canonical ID column"""

from __future__ import annotations

import warnings

import pandas as pd
import polars as pl
import pytest

import liken as lk
from liken._constants import CANONICAL_ID


# SET UP:

SCHEMA = ["uid", "address"]
SINGLE_COL = "address"


def build_pipeline_global(preprocessors):
    return lk.rules.pipeline(preprocessors=preprocessors).step(
        lk.rules.on(SINGLE_COL).exact()
    )


def build_pipeline_step(preprocessors):
    return lk.rules.pipeline().step(
        lk.rules.on(SINGLE_COL).exact(), preprocessors=preprocessors
    )


def build_pipeline_both(preprocessors):
    return lk.rules.pipeline(preprocessors=preprocessors).step(
        lk.rules.on(SINGLE_COL, preprocessors=preprocessors).exact()
    )


PIPELINE_BUILDERS = [
    build_pipeline_global,
    build_pipeline_step,
    build_pipeline_both,
]

PARAMS = [
    # "STRIP"
    ([], [[0, "   123ab, OL5 "], [1, "123ab, OL5"]], [0, 1]),
    ([lk.preprocessors.strip()], [[0, "   123ab, OL5 "], [1, "123ab, OL5"]], [0, 0]),
    # "LOWER"
    ([], [[0, "123AB, OL5"], [1, "123ab, OL5"]], [0, 1]),
    ([lk.preprocessors.lower()], [[0, "123AB, OL5"], [1, "123ab, OL5"]], [0, 0]),
    # "ALNUM"
    ([], [[0, "123ab, OL5"], [1, "123ab, OL5!!!"]], [0, 1]),
    ([lk.preprocessors.alnum()], [[0, "123ab, OL5"], [1, "123ab, OL5!!!"]], [0, 0]),
    # "REMOVE PUNCTUATION"
    ([], [[0, "123ab, OL5, UK"], [1, "123ab OL5 UK"]], [0, 1]),
    ([lk.preprocessors.remove_punctuation()], [[0, "123ab, OL5, UK"], [1, "123ab OL5 UK"]], [0, 0]),
    # "NORMALIZE UNICODE"
    ([], [[0, "Râñdòm Stréèt"], [1, "Râñdòm Stréèt"]], [0, 1]),
    ([lk.preprocessors.normalize_unicode()], [[0, "Râñdòm Stréèt"], [1, "Râñdòm Stréèt"]], [0, 0]),
    # "ASCII FOLD"
    ([], [[0, "Râñdòm Stréèt"], [1, "Random Street"]], [0, 1]),
    ([lk.preprocessors.ascii_fold()], [[0, "Râñdòm Stréèt"], [1, "Random Street"]], [0, 0]),
    # "REMOVE STOPWORDS"
    ([], [[0, "this is a Random Street"], [1, "   Random Street"]], [0, 1]),
    ([lk.preprocessors.remove_stopwords()], [[0, "this is a Random Street"], [1, "   Random Street"]], [0, 0]),
    # "NORMALIZE NAMES"
    ([], [[0, "Mr. John H Doe (Da Legend)"], [1, "John H Doe"]], [0, 1]),
    ([lk.preprocessors.normalize_names()], [[0, "Mr. John H Doe (Da Legend)"], [1, "John H Doe"]], [0, 0]),
    # "NORMALIZE COMPANY"
    ([], [[0, "Random Services LLC."], [1, "Random Services"]], [0, 1]),
    ([lk.preprocessors.normalize_company()], [[0, "Random Services LLC."], [1, "Random Services"]], [0, 0]),
]
IDS = [
    "strip-void",
    "strip-dedupes",
    "lower-void",
    "lower-dedupes",
    "alnum-void",
    "alnum-dedupes",
    "remove-punctuation-void",
    "remove-punctuation-dedupes",
    "normalize-unicode-void",
    "normalize-unicode-dedupes",
    "ascii-fold-void",
    "ascii-fold-dedupes",
    "remove-stopwords-void",
    "remove-stopwords-dedupes",
    "normalize-names-void",
    "normalize-names-dedupes",
    "normalize-company-void",
    "normalize-company-dedupes",
]


@pytest.mark.parametrize("preprocessors, data, expected_canonical_id", PARAMS, ids=IDS)
@pytest.mark.parametrize("backend", ["pandas", "polars", "spark"])
@pytest.mark.parametrize("pipeline_builder", PIPELINE_BUILDERS)
def test_preprocessors_from_pipeline(
    backend,
    preprocessors,
    data,
    expected_canonical_id,
    pipeline_builder,
    spark,
    helpers,
):
    pipeline = pipeline_builder(preprocessors)

    df = create_df(backend, spark, data)
    df = dedupe_df(df, spark, pipeline)

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_canonical_id


# HELPERS:


def create_df(backend, spark, data):
    if backend == "pandas":
        df = pd.DataFrame(columns=SCHEMA, data=data)

    if backend == "polars":
        df = pl.DataFrame(schema=SCHEMA, data=data, orient="row")

    if backend == "spark":
        df = spark.createDataFrame(schema=SCHEMA, data=data)
    return df


def dedupe_df(df, spark, pipeline):
    return lk.dedupe(df, spark_session=spark).apply(pipeline).canonicalize().collect()
