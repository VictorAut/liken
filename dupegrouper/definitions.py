"""constants and types"""

from __future__ import annotations
import os
import typing

import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, Row
from pyspark.sql.types import (
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    FloatType,
    BooleanType,
    TimestampType,
    DateType,
)

if typing.TYPE_CHECKING:
    from dupegrouper.strategies import BaseStrategy  # pragma: no cover


# CONSTANTS


# the canonical_id label in the dataframe
CANONICAL_ID: typing.Final[str] = os.environ.get("CANONICAL_ID", "canonical_id")


# TYPES:


StrategyMapCollection: typing.TypeAlias = typing.DefaultDict[
    str,
    list["BaseStrategy | tuple[typing.Callable, dict[str, str]]"],
]


DataFrameLike: typing.TypeAlias = "pd.DataFrame | pl.DataFrame | SparkDataFrame | list[Row]"  # | ...
SeriesLike: typing.TypeAlias = "pd.Series | pl.Series | list[typing.Any]"  # | ...


# PYSPARK SQL TYPES TO CLASS TYPE CONVERSION


PYSPARK_TYPES = {
    "string": StringType(),
    "int": IntegerType(),
    "bigint": LongType(),
    "double": DoubleType(),
    "float": FloatType(),
    "boolean": BooleanType(),
    "timestamp": TimestampType(),
    "date": DateType(),
    # ...
}
