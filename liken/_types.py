"""Shared liken types"""

from __future__ import annotations

from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import pyspark.sql as spark


# TYPES:

type UserDataFrame = pd.DataFrame | pl.DataFrame | spark.DataFrame
type DataFrameLike = pd.DataFrame | pl.DataFrame | spark.DataFrame | list[spark.Row]
type SeriesLike = pd.Series | pl.Series | list[Any]
type ArrayLike = np.ndarray | pd.Series | pl.Series | list[Any]
type Columns = str | tuple[str, ...]  # label(s) that identify attributes of a dataframe for deduplication
type Keep = Literal["first", "last"]  # Canonicalisation rule
type SimilarPairIndices = tuple[int, int]
