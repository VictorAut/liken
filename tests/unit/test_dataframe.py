from __future__ import annotations

from unittest.mock import create_autospec

from pyspark.sql import DataFrame as SparkDataFrame, Row
import pandas as pd
import polars as pl
import pytest

import numpy as np

from dupegrouper.dataframe import (
    PandasDF,
    PolarsDF,
    SparkDF,
    SparkRows,
    wrap,
)


# FIXTURES:


@pytest.fixture
def mock_df_spark():
    return create_autospec(SparkDataFrame)


@pytest.fixture
def spark_rows():
    row1 = Row(a=1, b=2)
    row2 = Row(a=3, b=4)
    return [row1, row2]


@pytest.fixture
def new_col():
    return ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]


# wrapper methods


def test_wrapper_methods_pandas(df_pandas, new_col):
    wdf = PandasDF(df_pandas)

    result = wdf.put_col("new_col", new_col)
    assert result is wdf
    assert "new_col" in wdf.unwrap().columns

    series = wdf.get_col("new_col")
    assert isinstance(series, pd.Series)

    df_subset = wdf.get_cols(("email", "account"))
    assert isinstance(df_subset, pd.DataFrame)
    assert list(df_subset.columns) == ["email", "account"]


def test_wrapper_methods_polars(df_polars, new_col):
    wdf = PolarsDF(df_polars)

    result = wdf.put_col("test_col", new_col)
    assert result is wdf
    assert "test_col" in wdf.unwrap().columns

    series = wdf.get_col("test_col")
    assert hasattr(series, "dtype")  # basic polars Series check

    df_subset = wdf.get_cols(("email", "account"))
    assert hasattr(df_subset, "columns")


def test_wrapper_methods_spark(df_spark):
    wdf = SparkDF(df_spark)

    with pytest.raises(NotImplementedError):
        wdf.put_col()
    with pytest.raises(NotImplementedError):
        wdf.get_col()
    with pytest.raises(NotImplementedError):
        wdf.get_cols()


def test_wrapper_methods_sparkrows(df_sparkrows, new_col):
    wdf = SparkRows(df_sparkrows)

    result = wdf.put_col("new_col", new_col)
    assert result is wdf

    col_values = wdf.get_col("new_col")
    assert col_values == new_col

    cols_values = wdf.get_cols(("email", "account"))
    assert all(isinstance(c, list) for c in cols_values)


# DataFrame delegation


def test_frame_getattr_delegates(df_pandas):
    wdf = PandasDF(df_pandas)
    # `head` is a pd.DataFrame method
    assert wdf.head == wdf._df.head


# wrap dispatcher


def test_wrap_dispatch(df_pandas, df_polars, df_spark, df_sparkrows):
    # Pandas
    wrapped = wrap(df_pandas, id="id")
    assert isinstance(wrapped, PandasDF)
    # Polars
    wrapped = wrap(df_polars, id="id")
    assert isinstance(wrapped, PolarsDF)
    # Spark DataFrame
    wrapped = wrap(df_spark, id="id")
    assert isinstance(wrapped, SparkDF)
    # List of Rows
    wrapped = wrap(df_sparkrows, id="id")
    assert isinstance(wrapped, SparkRows)
    # Unsupported type
    with pytest.raises(NotImplementedError, match="Unsupported data frame"):
        wrap("not_a_df")


# Add Canonical ID


@pytest.mark.parametrize("backend", ["pandas", "polars", "spark"])
def test_id_matrix(backend, spark):

    if backend == "pandas":
        df = pd.DataFrame(columns=["test", "hello"])

    if backend == "polars":
        df = pl.DataFrame(schema=["test", "hello"])

    if backend == "spark":
        df = spark.createDataFrame(schema=["test", "hello"], data=[])



from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

#Creates Empty RDD
emptyRDD = spark.sparkContext.emptyRDD()

from pyspark.sql.types import StructType,StructField, StringType
schema = StructType([
  StructField('firstname', StringType(), True),
  StructField('middlename', StringType(), True),
  StructField('lastname', StringType(), True)
  ])

spark.createDataFrame(emptyRDD,schema=["hello", "bye"])