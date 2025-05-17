from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dupegrouper.definitions import GROUP_ID
from dupegrouper.wrappers.dataframes import WrappedSparkDataFrame, WrappedPandasDataFrame, WrappedPolarsDataFrame


def test__add_group_id(lowlevel_dataframe, helpers):

    df, wrapper, id = lowlevel_dataframe

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        assert GROUP_ID not in df.unwrap().columns  # top level Spark Dataframe wrapper has NO implementations
    else:
        assert helpers.get_column_as_list(df.unwrap(), GROUP_ID) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


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


def test_get_col(lowlevel_dataframe, group_id):
    df, wrapper, id = lowlevel_dataframe

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        with pytest.raises(NotImplementedError, match=df.not_implemented):
            df.get_col()
    else:
        assert list(df.get_col(GROUP_ID)) == group_id


@pytest.mark.parametrize(
    "mapping, output",
    [
        (
            {"123ab, OL5 9PL, UK": "test"},
            ["test", None, None, None, None, None, None, None, "test", "test", None, None, None],
        ),
        (
            {
                "123ab, OL5 9PL, UK": "map_1",
                "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom": "map_2",
                "Calle Ancho, 12, 05688, Rioja, Navarra, Espana": "map_3",
                "Calle Sueco, 56, 05688, Rioja, Navarra": "map_4",
                "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom": "map_5",
                "66b Porters street, OL5 9PL, Newark, United Kingdom": "map_6",
                "C. Ancho 49, 05687, Navarra": "map_6",
                "Ambleside avenue Park Road ED3, UK": "map_8",
                "123ab, OL5 9PL, UK": "map_9",  # noqa: F601
                "123ab, OL5 9PL, UK": "map_10",  # noqa: F601
                "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK": "map_11",
                "37 GH9, UK": "map_12",
                "totally random non existant address": "map_13",
            },
            [
                "map_10",
                "map_2",
                "map_3",
                "map_4",
                "map_5",
                "map_6",
                "map_6",
                "map_8",
                "map_10",
                "map_10",
                "map_11",
                "map_12",
                "map_13",
            ],
        ),
        (
            {
                # "123ab, OL5 9PL, UK": "map_1",
                "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom": "map_2",
                "Calle Ancho, 12, 05688, Rioja, Navarra, Espana": "map_3",
                "Calle Sueco, 56, 05688, Rioja, Navarra": "map_4",
                "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom": "map_5",
                "66b Porters street, OL5 9PL, Newark, United Kingdom": "map_6",
                "C. Ancho 49, 05687, Navarra": "map_6",
                "Ambleside avenue Park Road ED3, UK": "map_8",
                # "123ab, OL5 9PL, UK": "map_9",
                "123ab, OL5 9PL, UK": "map_10",
                "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK": "map_11",
                "37 GH9, UK": "map_12",
                "totally random non existant address": "map_13",
            },
            [
                "map_10",
                "map_2",
                "map_3",
                "map_4",
                "map_5",
                "map_6",
                "map_6",
                "map_8",
                "map_10",
                "map_10",
                "map_11",
                "map_12",
                "map_13",
            ],
        ),
    ],
)
def test_map_dict(mapping, output, lowlevel_dataframe):
    df, wrapper, id = lowlevel_dataframe

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        with pytest.raises(NotImplementedError, match=df.not_implemented):
            df.map_dict()
    elif isinstance(df, WrappedPandasDataFrame):
        output = [x if x else np.nan for x in output]
        assert list(df.map_dict("address", mapping)) == output
    elif isinstance(df, WrappedPolarsDataFrame):
        assert list(df.map_dict("address", mapping)) == output
    else:
        assert df.map_dict("address", mapping) == output


def test_drop_col(lowlevel_dataframe):
    df, wrapper, id = lowlevel_dataframe

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        with pytest.raises(NotImplementedError, match=df.not_implemented):
            df.drop_col()
    elif isinstance(df, WrappedPandasDataFrame) or isinstance(df, WrappedPolarsDataFrame):
        df.drop_col("address")
        assert "address" not in df.unwrap().columns
    else:
        df.drop_col("address")
        for row in df.unwrap():
            assert "address" not in row.asDict().keys()


@pytest.mark.parametrize(
    "series, array, output",
    # string types
    [
        (
            ["test", None, None, None, None, None, None, None, "test", "test", None, None, None],
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
            ["test", "b", "c", "d", "e", "f", "g", "h", "test", "test", "k", "l", "m"],
        ),
        (
            ["test", None, None, None, None, None, None, None, "test", "test", None, None, None],
            ["a", "b", None, "d", "e", "f", "g", "h", "i", "j", "k", "l", None],
            ["test", "b", None, "d", "e", "f", "g", "h", "test", "test", "k", "l", None],
        ),
    ],
)
def test_fill_na(series, array, output, lowlevel_dataframe):
    df, wrapper, id = lowlevel_dataframe

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        with pytest.raises(NotImplementedError, match=df.not_implemented):
            df.fill_na()
    elif isinstance(df, WrappedPandasDataFrame):
        series = pd.Series(series)
        array = pd.Series(array)
        assert list(df.fill_na(series, array)) == output

    elif isinstance(df, WrappedPolarsDataFrame):
        series = pl.Series(series)
        array = pl.Series(array)
        assert list(df.fill_na(series, array)) == output

    else:
        assert df.fill_na(series, array) == output
