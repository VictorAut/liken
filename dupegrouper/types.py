from __future__ import annotations
from collections import defaultdict, UserDict
from typing import Any, Literal, TypeAlias, TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, Row

if TYPE_CHECKING:
    from dupegrouper.strats import BaseStrategy  # pragma: no cover


# TYPES:


StrategyMapCollection: TypeAlias = defaultdict[
    str,
    list["BaseStrategy"],
]

class StrNumberDict(UserDict):
    def __setitem__(self, key, value):
        if not isinstance(key, (str, tuple)):
            raise TypeError(
                f'Invalid type for dictionary key: '
                f'expected "str" of "tuple", got "{type(key).__name__}"'
            )
        if not isinstance(key, list | tuple): # TODO: iterable?
            raise TypeError(
                f'Invalid type for dictionary value: '
                f'expected "Iterable", got "{type(value).__name__}"'
            )
        if not all(isinstance(instance, BaseStrategy) for instance in value):
            raise TypeError(
                f'Invalid type for dictionary value: '
                f'expected "Number", got "{type(value).__name__}"'
            )
        super().__setitem__(key, value)

DataFrameLike: TypeAlias = "pd.DataFrame | pl.DataFrame | SparkDataFrame | list[Row]"

SeriesLike: TypeAlias = "pd.Series | pl.Series | list[Any]"

ArrayLike: TypeAlias = "np.ndarray | pd.Series | pl.Series | list[Any]"

# label(s) that identify attributes of a dataframe that are to be deduplicated
Columns: TypeAlias = str | tuple[str, ...]

# Canonicalisation rule
Rule: TypeAlias = Literal["first", "last"]

SimilarPairIndices: TypeAlias = tuple[int, int]