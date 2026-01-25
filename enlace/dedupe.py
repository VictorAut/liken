"""enlace main entrypoint

This module contains `Dedupe`, at the core of all 'dupe and group'
functionality provided by enlace.
"""

from __future__ import annotations

from typing import Generic
from typing import overload

import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.sql import SparkSession

from enlace import exact
from enlace._dataframe import DF
from enlace._dataframe import wrap
from enlace._executors import Executor
from enlace._executors import LocalExecutor
from enlace._executors import SparkExecutor
from enlace._strats_library import BaseStrategy
from enlace._strats_manager import Rules
from enlace._strats_manager import StrategyManager
from enlace._strats_manager import StratsDict
from enlace._types import Columns
from enlace._types import DataFrameLike
from enlace._types import Keep
from enlace._validators import validate_columns_arg
from enlace._validators import validate_keep_arg
from enlace._validators import validate_spark_args


# API:


class Dedupe(Generic[DF]):
    """TODO"""

    _df: DF
    _executor: Executor[DF]

    @overload
    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame | list[spark.Row],
        /,
        *,
        spark_session: None = None,
        id: str | None = None,
    ): ...

    @overload
    def __init__(
        self,
        df: spark.DataFrame,
        /,
        *,
        spark_session: SparkSession,
        id: str,
    ): ...

    def __init__(
        self,
        df: DataFrameLike,
        /,
        *,
        spark_session: SparkSession | None = None,
        id: str | None = None,
    ):
        self._sm = StrategyManager()

        self._executor: LocalExecutor | SparkExecutor
        if isinstance(df, spark.DataFrame):
            spark_session = validate_spark_args(spark_session)
            self._executor = SparkExecutor(spark_session=spark_session, id=id)
        else:
            self._executor = LocalExecutor()

        self._df = wrap(df, id)

    def apply(self, strategy: BaseStrategy | StratsDict | dict) -> None:
        self._sm.apply(strategy)

    def canonicalize(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
        drop_duplicates: bool = False,
    ) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
        """canonicalize, and group, the data based on the provided attribute

        Args:
            columns: The attribute to deduplicate. If strategies have been added
                as a mapping object, this must not passed, as the keys of the
                mapping object will be used instead
        """
        keep = validate_keep_arg(keep)
        columns = validate_columns_arg(columns, self._sm.is_sequential_applied)

        # Allow use of no .apply(), assuming exact deduplication
        if not self._sm._has_applies:
            self.apply(exact())
        strats: StratsDict | Rules = self._sm.get()

        self._df = self._executor.execute(
            self._df,
            columns=columns,
            strats=strats,
            keep=keep,
            drop_duplicates=drop_duplicates,
            drop_canonical_id=False,
        )

        self._sm.reset()

        return self._df.unwrap()

    def drop_duplicates(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
    ) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
        """canonicalize, and group, the data based on the provided attribute

        Args:
            columns: The attribute to deduplicate. If strategies have been added
                as a mapping object, this must not passed, as the keys of the
                mapping object will be used instead
        """
        keep = validate_keep_arg(keep)
        columns = validate_columns_arg(columns, self._sm.is_sequential_applied)

        # Allow use of no .apply(), assuming exact deduplication
        if not self._sm._has_applies:
            self._sm.apply(exact())
        strats: StratsDict | Rules = self._sm.get()

        self._df = self._executor.execute(
            self._df,
            columns=columns,
            strats=strats,
            keep=keep,
            drop_duplicates=True,
            drop_canonical_id=True,
        )

        self._sm.reset()

        return self._df.unwrap()

    @property
    def strats(self) -> str | None:
        """
        Returns the strategies currently stored in the strategy manager.

        If no strategies are stored, returns None. Otherwise, returns a string
        representation of the strategies containedtuple
        of strategy names or a dictionary mapping attributes to their
        respective strategies.

        Returns:
            The stored strategies, formatted
        """
        return self._sm.pretty_get()

    @property
    def df(self) -> DataFrameLike:
        return self._df.unwrap()
