from __future__ import annotations
import typing

import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, Row

if typing.TYPE_CHECKING:
    from dupegrouper.strats import BaseStrategy  # pragma: no cover


# TYPES:


StrategyMapCollection: typing.TypeAlias = typing.DefaultDict[
    str,
    list["BaseStrategy | tuple[typing.Callable, dict[str, str]]"],
]


DataFrameLike: typing.TypeAlias = "pd.DataFrame | pl.DataFrame | SparkDataFrame | list[Row]"  # | ...
SeriesLike: typing.TypeAlias = "pd.Series | pl.Series | list[typing.Any]"  # | ...
