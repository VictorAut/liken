"""Shared liken types"""

from __future__ import annotations

from typing import Literal
from typing import TypeAlias

import modin.pandas as mpd
import pandas as pd
import polars as pl
import pyspark.sql as spark
from ray.data import Dataset as RayDataset


# TODO: `DataFrameLike` needs to be UserDataFrame | list[spark.row]
UserDataFrame: TypeAlias = pd.DataFrame | pl.DataFrame | mpd.DataFrame | RayDataset | spark.DataFrame
DataFrameLike: TypeAlias = pd.DataFrame | pl.DataFrame | mpd.DataFrame | RayDataset | spark.DataFrame | list[spark.Row]
Columns: TypeAlias = str | tuple[str, ...]  # label(s) that identify attributes of a dataframe for deduplication
Keep: TypeAlias = Literal["first", "last"]  # Canonicalisation rule
SimilarPairIndices: TypeAlias = tuple[int, int]
