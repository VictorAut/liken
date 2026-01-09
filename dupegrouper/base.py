"""dupegrouper main entrypoint

This module contains `Duped`, at the core of all 'dupe and group'
functionality provided by dupegrouper.
"""

from __future__ import annotations
import logging
from typing import overload

import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.sql import SparkSession
from dupegrouper.dataframe import (
    wrap,
    WrappedDataFrame,
    WrappedSparkDataFrame,
)
from dupegrouper.executors import LocalExecutor, SparkExecutor
from dupegrouper.strats_library import BaseStrategy
from dupegrouper.strats_manager import StrategyManager
from dupegrouper.types import (
    Columns,
    DataFrameLike,
    Rule,
)


# LOGGER:


logger = logging.getLogger(__name__)


# BASE:


class Duped:
    """Top-level entrypoint for grouping duplicates

    This class handles initialisation of a dataframe, dispatching appropriately
    given the supported dataframe libraries (e.g. Pandas). An instance of this
    class can then accept a variety of strategies for deduplication and
    grouping.

    Upon initialisation, `Duped` sets a new column, usually `"canonical_id"`
    — but you can control this by setting an environment variable `CANONICAL_ID` at
    runtime. The canonical_id is a monotonically increasing, numeric id column
    starting at 1 to the length of the dataframe provided.
    """

    @overload
    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame,
        /,
        *,
        spark_session: None = None,
        id: str | None = None,
        keep: Rule = "first",
    ): ...

    @overload
    def __init__(
        self,
        df: spark.DataFrame,
        /,
        *,
        spark_session: SparkSession,
        id: str,
        keep: Rule = "first",
    ): ...

    def __init__(
        self,
        df: DataFrameLike,
        /,
        *,
        spark_session: SparkSession | None = None,
        id: str | None = None,
        keep: Rule = "first",
    ):
        self._df: WrappedDataFrame = wrap(df, id)
        self._sm = StrategyManager()

        if isinstance(self._df, WrappedSparkDataFrame):
            self._executor = SparkExecutor(
                keep=keep,
                spark_session=spark_session,
                id=id,
            )
        else:
            self._executor = LocalExecutor(keep=keep)

    def apply(self, strategy: BaseStrategy | dict) -> None:
        self._sm.apply(strategy)

    def canonicalize(self, columns: Columns | None = None) -> None:
        """canonicalize, and group, the data based on the provided attribute

        Args:
            columns: The attribute to deduplicate. If strategies have been added
                as a mapping object, this must not passed, as the keys of the
                mapping object will be used instead
        """
        strategies = self._sm.get()

        self._df = self._executor.canonicalize(self._df, columns, strategies)

        self._sm.reset()

    @property
    def strategies(self) -> None | tuple[str, ...] | dict[str, tuple[str, ...]]:
        """
        Returns the strategies currently stored in the strategy manager.

        If no strategies are stored, returns `None`. Otherwise, returns a tuple
        of strategy names or a dictionary mapping attributes to their
        respective strategies.

        Returns:
            The stored strategies, formatted
        """
        return self._sm.pretty_get()

    @property
    def df(self) -> DataFrameLike:
        return self._df.unwrap()
