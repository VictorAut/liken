from __future__ import annotations
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import polars as pl
import pyspark.sql as spark


# TYPES:


DataFrameLike: TypeAlias = "pd.DataFrame | pl.DataFrame | spark.DataFrame | list[spark.Row]"
SeriesLike: TypeAlias = "pd.Series | pl.Series | list[Any]"
ArrayLike: TypeAlias = "np.ndarray | pd.Series | pl.Series | list[Any]"
Columns: TypeAlias = str | tuple[str, ...] # label(s) that identify attributes of a dataframe for deduplication
Rule: TypeAlias = Literal["first", "last"] # Canonicalisation rule
SimilarPairIndices: TypeAlias = tuple[int, int]



from typing import Generic, TypeVar

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self) -> None:
        self._store: dict[str, T] = {}
          
    def set_item(self, k: str, v: T) -> None:
        self._store[k] = v
    
    def get_item(self, k: str) -> T:
        return self._store[k]
  

family_name_reg = Registry[str]()
family_age_reg = Registry[int]()

family_name_reg.set_item("husband", "steve")
family_name_reg.set_item("dad", "john")

family_age_reg.set_item("steve", 30)