from __future__ import annotations
from collections import defaultdict, UserDict
from typing import Any, Literal, TypeAlias, TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, Row

if TYPE_CHECKING:
    from dupegrouper.strats_library import BaseStrategy  # pragma: no cover

from dupegrouper.strats_library import BaseStrategy # TODO

# TYPES:


# StratsConfig: TypeAlias = defaultdict[
#     str,
#     list["BaseStrategy"],
# ]

class StratsConfig(UserDict):
    def __setitem__(self, key, value):
        if not isinstance(key, str | tuple):
            raise TypeError(
                f'Invalid type for strat dictionary key: '
                f'expected "str" of "tuple", got "{type(key).__name__}"'
            )
        if not isinstance(value, list | tuple):
            raise TypeError(
                f'Invalid type for strat dictionary value: '
                f'expected "list" or "tuple", got "{type(value).__name__}"'
            )
        if not all(isinstance(instance, BaseStrategy) for instance in value):
            raise TypeError(
                f'Invalid type for strat dictionary value: '
                f'expected "Number", got "{type(value).__name__}"'
            )
        super().__setitem__(key, value)

DataFrameLike: TypeAlias = "pd.DataFrame | pl.DataFrame | SparkDataFrame | list[Row]"
SeriesLike: TypeAlias = "pd.Series | pl.Series | list[Any]"
ArrayLike: TypeAlias = "np.ndarray | pd.Series | pl.Series | list[Any]"
Columns: TypeAlias = str | tuple[str, ...] # label(s) that identify attributes of a dataframe for deduplication
Rule: TypeAlias = Literal["first", "last"] # Canonicalisation rule
SimilarPairIndices: TypeAlias = tuple[int, int]