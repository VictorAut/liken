"""Shared liken types"""

from __future__ import annotations

from typing import Literal
from typing import TypeAlias

import dask.dataframe as dd
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pyspark.sql as spark
from ray.data import Dataset as RayDataset


SupportedBackends: TypeAlias = Literal["pandas", "polars", "modin", "spark", "ray", "dask"]
UserDataFrame: TypeAlias = pd.DataFrame | pl.DataFrame | mpd.DataFrame | RayDataset | dd.DataFrame | spark.DataFrame
InternalDataFrame: TypeAlias = UserDataFrame | list[spark.Row]
Columns: TypeAlias = str | tuple[str, ...]  # label(s) that identify attributes of a dataframe for deduplication
Keep: TypeAlias = Literal["first", "last"]  # Canonicalisation rule
SimilarPairIndices: TypeAlias = tuple[int, int]
