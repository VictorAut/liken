"""Shared liken types"""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import TypeAlias

import numpy as np
import pandas as pd
import polars as pl
import pyspark.sql as spark


DataFrameLike: TypeAlias = "pd.DataFrame | pl.DataFrame | spark.DataFrame | list[spark.Row]"
SeriesLike: TypeAlias = "pd.Series | pl.Series | list[Any]"
ArrayLike: TypeAlias = "np.ndarray | pd.Series | pl.Series | list[Any]"
Columns: TypeAlias = str | tuple[str, ...]  # label(s) that identify attributes of a dataframe for deduplication
Keep: TypeAlias = Literal["first", "last"]  # Canonicalisation rule
SimilarPairIndices: TypeAlias = tuple[int, int]
