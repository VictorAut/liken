from __future__ import annotations
import pytest
from unittest.mock import create_autospec
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame, Row
import numpy as np

from dupegrouper.dataframe import (
    PandasDF,
    PolarsDF,
    SparkDF,
    SparkRows,
    wrap,
    CANONICAL_ID,
)


# Fixtures


@pytest.fixture
def mock_spark_df():
    return create_autospec(SparkDataFrame)


@pytest.fixture
def spark_rows():
    row1 = Row(a=1, b=2)
    row2 = Row(a=3, b=4)
    return [row1, row2]


##################
# PandasDF tests #
##################


def test_wrapper_methods_pandas(df_pandas, marital_status):
    df_wrapper = PandasDF(df_pandas, id="uid")
    assert CANONICAL_ID in df_wrapper.unwrap().columns
    assert CANONICAL_ID in df_wrapper.columns  # Note delegation layer!

    result = df_wrapper.put_col("test_col", marital_status)
    assert result is df_wrapper
    assert "test_col" in df_wrapper.unwrap().columns

    series = df_wrapper.get_col("test_col")
    assert isinstance(series, pd.Series)

    df_subset = df_wrapper.get_cols(("email", "account"))
    assert isinstance(df_subset, pd.DataFrame)
    assert list(df_subset.columns) == ["email", "account"]


##################
# PolarsDF tests #
##################


def test_wrapper_methods_polars(df_polars, marital_status):
    df_wrapper = PolarsDF(df_polars, id="uid")
    assert CANONICAL_ID in df_wrapper.unwrap().columns
    assert CANONICAL_ID in df_wrapper.columns  # Note delegation layer!

    result = df_wrapper.put_col("test_col", marital_status)
    assert result is df_wrapper
    assert "test_col" in df_wrapper.unwrap().columns

    series = df_wrapper.get_col("test_col")
    assert hasattr(series, "dtype")  # basic polars Series check

    df_subset = df_wrapper.get_cols(("email", "account"))
    assert hasattr(df_subset, "columns")


#################
# SparkDF tests #
#################


def test_wrapper_methods_spark(mock_spark_df):
    df_wrapper = SparkDF(mock_spark_df, id="uid")
    with pytest.raises(NotImplementedError):
        df_wrapper._add_canonical_id()
    with pytest.raises(NotImplementedError):
        df_wrapper.put_col()
    with pytest.raises(NotImplementedError):
        df_wrapper.get_col()
    with pytest.raises(NotImplementedError):
        df_wrapper.get_cols()


###################
# SparkRows tests #
###################


def test_wrapper_methods_sparkrows(df_sparkrows, marital_status):
    df_wrapper = SparkRows(df_sparkrows, id="id")
    for row in df_wrapper.unwrap():
        assert CANONICAL_ID not in row.asDict()

    result = df_wrapper.put_col("new_col", marital_status)
    assert result is df_wrapper

    col_values = df_wrapper.get_col("new_col")
    assert col_values == marital_status

    cols_values = df_wrapper.get_cols(("email", "account"))
    assert all(isinstance(c, list) for c in cols_values)
    assert len(cols_values) == len(df_sparkrows[0].asDict())


####################
# Frame delegation #
####################


def test_frame_getattr_delegates(df_pandas):
    df_wrapper = PandasDF(df_pandas, id="uid")
    # 'head' is a method of pd.DataFrame
    head_method = df_wrapper.head
    assert callable(head_method)


###################
# wrap dispatcher #
###################


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
    with pytest.raises(NotImplementedError):
        wrap("not_a_df")


#################
# put_col tests #
#################


@pytest.mark.parametrize(
    "array",
    # string types
    [
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
        # numeric types
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ],
)
def test_put_col(array, lowlevel_dataframe, helpers):
    df, wrapper, id = lowlevel_dataframe

    df = wrapper(df, id)

    if isinstance(df, SparkDF):
        with pytest.raises(NotImplementedError):
            df.put_col()
    else:
        df.put_col("TEST", np.array(array))
        assert helpers.get_column_as_list(df.unwrap(), "TEST") == array
