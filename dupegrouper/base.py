"""dupegrouper main entrypoint

This module contains `Duped`, at the core of all 'dupe and group'
functionality provided by dupegrouper.
"""

from __future__ import annotations
from collections.abc import Iterator
from functools import singledispatchmethod
import logging
from types import NoneType
from typing import cast

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructField, StructType, DataType

from dupegrouper.constants import (
    CANONICAL_ID,
    DEFAULT_STRAT_KEY,
    PYSPARK_TYPES,
)
from dupegrouper.dataframe import (
    wrap,
    WrappedDataFrame,
    WrappedSparkDataFrame,
)
from dupegrouper.executors import LocalExecutor, SparkExecutor
from dupegrouper.strats_library import BaseStrategy
from dupegrouper.strats_manager import StrategyManager, StratsConfig
from dupegrouper.types import (
    Columns,
    DataFrameLike,
    Rule,
)


# LOGGER:


_logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        df: DataFrameLike,
        spark_session: SparkSession | None = None,
        id: str | None = None,
        canonicalization_rule: Rule = "first",
    ):
        self._df: WrappedDataFrame = wrap(df, id)
        self._sm = StrategyManager()
        self._spark_session = spark_session
        self._id = id
        self._canonicalization_rule = canonicalization_rule
        # TODO: validate that if ._df is spark then need a spark session.
        if isinstance(self._df, WrappedSparkDataFrame):
            self._executor = SparkExecutor(
                canonicalization_rule=canonicalization_rule,
                spark_session=spark_session,
                id=id,
            )
        else:
            self._executor = LocalExecutor(
                canonicalization_rule=canonicalization_rule,
            )

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
