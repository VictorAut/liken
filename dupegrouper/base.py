"""dupegrouper main entrypoint

This module contains `Duped`, at the core of all 'dupe and group'
functionality provided by dupegrouper.
"""

from __future__ import annotations

from typing import Generic, Literal, overload

import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.sql import SparkSession

from dupegrouper.dataframe import DF, wrap
from dupegrouper.executors import Executor, LocalExecutor, SparkExecutor
from dupegrouper.strats_library import BaseStrategy
from dupegrouper.strats_manager import StrategyManager, StratsConfig
from dupegrouper.types import Columns, DataFrameLike, Keep



# API:


class Duped(Generic[DF]):
    _df: DF
    _executor: Executor[DF]
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
        df: pd.DataFrame | pl.DataFrame | list[spark.Row],
        /,
        *,
        spark_session: None = None,
        id: str | None = None,
        keep: Keep = "first",
    ): ...

    @overload
    def __init__(
        self,
        df: spark.DataFrame,
        /,
        *,
        spark_session: SparkSession,
        id: str,
        keep: Keep = "first",
    ): ...

    def __init__(
        self,
        df: DataFrameLike,
        /,
        *,
        spark_session: SparkSession | None = None,
        id: str | None = None,
        keep: Keep = "first",
    ):
        self._sm = StrategyManager()

        # TODO: warnings e.g. if canonical_id is already present in the data

        keep = _validate_keep_arg(keep)

        self._executor: LocalExecutor | SparkExecutor
        if isinstance(df, spark.DataFrame):
            spark_session = _validate_spark_args(spark_session)
            self._executor = SparkExecutor(
                keep=keep,
                spark_session=spark_session,
                id=id,
            )
        else:
            self._executor = LocalExecutor(keep=keep)

        self._df = wrap(df, id)

    def apply(self, strategy: BaseStrategy | StratsConfig | dict) -> None:
        self._sm.apply(strategy)

    def canonicalize(self, columns: Columns | None = None) -> None:
        """canonicalize, and group, the data based on the provided attribute

        Args:
            columns: The attribute to deduplicate. If strategies have been added
                as a mapping object, this must not passed, as the keys of the
                mapping object will be used instead
        """
        strats = self._sm.get()

        self._df = self._executor.canonicalize(self._df, columns, strats)

        self._sm.reset()

    @property
    def strats(self) -> None | tuple[str, ...] | dict[str, tuple[str, ...]]:
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


def _validate_spark_args(spark_session: SparkSession | None = None, /) -> SparkSession:
    if not spark_session:
        raise ValueError("spark_session must be provided for a spark dataframe")
    return spark_session


def _validate_keep_arg(keep: Literal["first", "last"]) -> Literal["first", "last"]:
    if keep not in ("first", "last"):
        raise ValueError("Keep must be one of 'first' or 'last'")
    return keep
