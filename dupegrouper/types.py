from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable, Literal, TypeAlias, TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, Row

if TYPE_CHECKING:
    from dupegrouper.strats import BaseStrategy  # pragma: no cover


# TYPES:


StrategyMapCollection: TypeAlias = defaultdict[
    str,
    list["BaseStrategy | tuple[Callable, dict[str, str]]"],
]

DataFrameLike: TypeAlias = "pd.DataFrame | pl.DataFrame | SparkDataFrame | list[Row]"

SeriesLike: TypeAlias = "pd.Series | pl.Series | list[Any]"

ArrayLike: TypeAlias = "np.ndarray | pd.Series | pl.Series | list[Any]"

# label(s) that identify attributes of a dataframe that are to be deduplicated
Columns: TypeAlias = str | tuple[str, ...]

# Canonicalisation rule
Rule: TypeAlias = Literal["first", "last"]

SimilarPairIndices: TypeAlias = tuple[int, int]