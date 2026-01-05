from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dupegrouper.definitions import CANONICAL_ID
from dupegrouper.dataframe import (
    WrappedSparkDataFrame,
    WrappedPandasDataFrame,
    WrappedPolarsDataFrame,
)


def test__add_canonical_id(lowlevel_dataframe, helpers):

    df, wrapper, id = lowlevel_dataframe

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        assert CANONICAL_ID not in df.unwrap().columns  # top level Spark Dataframe wrapper has NO implementations
    else:
        assert helpers.get_column_as_list(df.unwrap(), CANONICAL_ID) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


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

    if isinstance(df, WrappedSparkDataFrame):
        with pytest.raises(NotImplementedError, match=df.not_implemented):
            df.put_col()
    else:
        df.put_col("TEST", np.array(array))
        assert helpers.get_column_as_list(df.unwrap(), "TEST") == array


def test_get_col(lowlevel_dataframe, canonical_id):
    df, wrapper, id = lowlevel_dataframe

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        with pytest.raises(NotImplementedError, match=df.not_implemented):
            df.get_col()
    else:
        assert list(df.get_col(CANONICAL_ID)) == canonical_id
