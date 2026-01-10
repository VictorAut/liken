from __future__ import annotations
import pytest
from unittest.mock import Mock, create_autospec
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame, Row
import numpy as np

from dupegrouper.dataframe import (
    Frame,
    PandasDF,
    PolarsDF,
    SparkDF,
    SparkRows,
    wrap,
    CANONICAL_ID,
)


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def pd_df():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

@pytest.fixture
def pl_df():
    return pl.DataFrame({"a": [1, 2], "b": [3, 4]})

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

def test_pandas_add_canonical_id(pd_df):
    df_wrapper = PandasDF(pd_df, id="uid")
    assert CANONICAL_ID in df_wrapper.unwrap().columns
    # put_col returns self and sets column
    result = df_wrapper.put_col("c", [9, 8])
    assert result is df_wrapper
    assert "c" in df_wrapper.unwrap().columns
    # get_col returns a Series
    series = df_wrapper.get_col("a")
    assert isinstance(series, pd.Series)
    # get_cols returns DataFrame
    df_subset = df_wrapper.get_cols(("a", "b"))
    assert isinstance(df_subset, pd.DataFrame)
    assert list(df_subset.columns) == ["a", "b"]


##################
# PolarsDF tests #
##################

def test_polarsdf_add_canonical_id(pl_df):
    df_wrapper = PolarsDF(pl_df, id="uid")
    assert CANONICAL_ID in df_wrapper.unwrap().columns
    # put_col returns self and sets column
    result = df_wrapper.put_col("c", [9, 8])
    assert result is df_wrapper
    assert "c" in df_wrapper.unwrap().columns
    # get_col returns pl.Series
    series = df_wrapper.get_col("a")
    assert hasattr(series, "dtype")  # basic polars Series check
    # get_cols returns DataFrame
    df_subset = df_wrapper.get_cols(("a", "b"))
    assert hasattr(df_subset, "columns")


#################
# SparkDF tests #
#################

def test_sparkdf_methods_raise(mock_spark_df):
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

def test_sparkrows_add_canonical_id_and_put_get(spark_rows):
    df_wrapper = SparkRows(spark_rows, id="a")
    # Check canonical id added
    for row in df_wrapper.unwrap():
        assert CANONICAL_ID in row.asDict()
    # put_col modifies rows and returns self
    new_values = [100, 200]
    result = df_wrapper.put_col("new_col", new_values)
    assert result is df_wrapper
    # get_col returns list
    col_values = df_wrapper.get_col("new_col")
    assert col_values == new_values
    # get_cols returns list of lists
    cols_values = df_wrapper.get_cols(("a", "b"))
    assert all(isinstance(c, list) for c in cols_values)
    assert len(cols_values) == len(spark_rows[0].asDict())


####################
# Frame delegation #
####################

def test_frame_getattr_delegates(pd_df):
    df_wrapper = PandasDF(pd_df, id="uid")
    # 'head' is a method of pd.DataFrame
    head_method = df_wrapper.head
    assert callable(head_method)


###################
# wrap dispatcher #
###################

def test_wrap_dispatch(pd_df, pl_df, mock_spark_df, spark_rows):
    # Pandas
    wrapped = wrap(pd_df, id="uid")
    assert isinstance(wrapped, PandasDF)
    # Polars
    wrapped = wrap(pl_df, id="uid")
    assert isinstance(wrapped, PolarsDF)
    # Spark DataFrame
    wrapped = wrap(mock_spark_df, id="uid")
    assert isinstance(wrapped, SparkDF)
    # List of Rows
    wrapped = wrap(spark_rows, id="a")
    assert isinstance(wrapped, SparkRows)
    # Unsupported type
    with pytest.raises(NotImplementedError):
        wrap("not_a_df")


#################
# put_col tests #
#################

def test_sparkrows_put_col_with_np_generic(spark_rows):
    df_wrapper = SparkRows(spark_rows, id="a")
    values = [np.int32(10), np.int32(20)]
    df_wrapper.put_col("new_col", values)
    col_values = df_wrapper.get_col("new_col")
    assert col_values == [10, 20]

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


####################################
# get_col tests i.e. single column #
####################################

def test_get_col(lowlevel_dataframe, canonical_id):
    df, wrapper, id = lowlevel_dataframe
    df = wrapper(df, id)

    if isinstance(df, SparkDF):
        with pytest.raises(NotImplementedError):
            df.get_col()
    else:
        assert list(df.get_col(CANONICAL_ID)) == canonical_id

######################################
# get_col tests i.e. multiple column #
######################################


# TODO:
# def test_get_cols(lowlevel_dataframe, account, birth_country):
#     df, wrapper, id = lowlevel_dataframe
#     df = wrapper(df, id)

#     compound = [list(i) for i in zip(account, birth_country)]

#     if isinstance(df, SparkDF):
#         with pytest.raises(NotImplementedError):
#             df.get_cols()
#     else:
#         assert list(df.get_cols(columns=("account", "birth_country"))) == compound