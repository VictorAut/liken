from __future__ import annotations

from unittest.mock import Mock, create_autospec

import pandas as pd
import pytest
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import Row

from dupegrouper.constants import CANONICAL_ID
from dupegrouper.dataframe import (
    CanonicalIdMixin,
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

    dropped = result.drop_col("new_col")
    assert dropped is wdf
    assert "new_col" not in wdf.unwrap().columns

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

    dropped = result.drop_col("test_col")
    assert dropped is wdf
    assert "test_col" not in wdf.unwrap().columns

    df_subset = wdf.get_cols(("email", "account"))
    assert hasattr(df_subset, "columns")


def test_wrapper_methods_spark(df_spark):
    wdf = SparkDF(df_spark, is_init=False)

    dropped = wdf.drop_col("address")
    assert dropped is wdf
    assert "address" not in wdf.unwrap().columns

    with pytest.raises(NotImplementedError):
        wdf.put_col()
    with pytest.raises(NotImplementedError):
        wdf.get_col()
    with pytest.raises(NotImplementedError):
        wdf.get_cols()
    with pytest.raises(NotImplementedError):
        wdf.drop_duplicates()


def test_wrapper_methods_sparkrows(df_sparkrows, new_col):
    wdf = SparkRows(df_sparkrows)

    result = wdf.put_col("new_col", new_col)
    assert result is wdf

    col_values = wdf.get_col("new_col")
    assert col_values == new_col

    cols_values = wdf.get_cols(("email", "account"))
    assert all(isinstance(c, list) for c in cols_values)


@pytest.fixture
def dummy_sparkrows():
    return [
        Row(canonical_id=1, address="a"),
        Row(canonical_id=1, address="b"),
        Row(canonical_id=2, address="c"),
        Row(canonical_id=2, address="d"),
    ]


def test_sparkrows_drop_duplicates_keep_first(dummy_sparkrows):

    # first

    wdf = SparkRows(dummy_sparkrows)

    deduped = wdf.drop_duplicates(keep="first")

    assert deduped is wdf
    assert [row.address for row in wdf._df] == ["a", "c"]

    # last

    wdf = SparkRows(dummy_sparkrows)

    deduped = wdf.drop_duplicates(keep="last")

    assert deduped is wdf
    assert [row.address for row in wdf._df] == ["b", "d"]


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


class DummyFrame(CanonicalIdMixin):
    def _df_as_is(self, df): ...
    def _df_overwrite_id(self, df, id): ...
    def _df_copy_id(self, df, id): ...
    def _df_autoincrement_id(self, df): ...


PARAMS = [
    (CANONICAL_ID, ["address", CANONICAL_ID], "_df_as_is"),
    ("uid", ["address", CANONICAL_ID], "_df_overwrite_id"),
    (None, ["address", CANONICAL_ID], "_df_as_is"),
    ("uid", ["address", "uid"], "_df_copy_id"),
    (None, ["address"], "_df_autoincrement_id"),
]
IDS = [
    "canonical id already exists; verbose definition",
    "new canonical id as overwrite from other id",
    "canonical id already exists; with warning",
    "new canonical id as write from other id",
    "new autoincremental canonical id",
]


@pytest.mark.filterwarnings("ignore:Canonical ID.*:UserWarning")
@pytest.mark.parametrize("id, cols, method", PARAMS, ids=IDS)
def test_add_canonical_id_mixin(id, cols, method):

    dummy = DummyFrame()
    dummy._df_as_is = Mock()
    dummy._df_overwrite_id = Mock()
    dummy._df_copy_id = Mock()
    dummy._df_autoincrement_id = Mock()

    df = Mock()
    df.columns = cols

    dummy._add_canonical_id(df, id)

    expected = getattr(dummy, method)
    expected.assert_called_once()


def test_add_canonical_id_warning():

    dummy = DummyFrame()
    dummy._df_as_is = Mock()

    df = Mock()
    df.columns = ["address", CANONICAL_ID]

    with pytest.warns(
        UserWarning,
        match=f"Canonical ID '{CANONICAL_ID}' already exists",
    ):
        dummy._add_canonical_id(df, id=None)

    dummy._df_as_is.assert_called_once()
