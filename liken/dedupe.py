"""liken main public API"""

from __future__ import annotations

from typing import Generic

import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.sql import SparkSession

from liken import exact
from liken._dataframe import DF
from liken._dataframe import Frame
from liken._dataframe import wrap
from liken._executors import Executor
from liken._executors import LocalExecutor
from liken._executors import SparkExecutor
from liken._strats_library import BaseStrategy
from liken._strats_manager import Rules
from liken._strats_manager import StrategyManager
from liken._strats_manager import StratsDict
from liken._types import Columns
from liken._types import Keep
from liken._validators import validate_columns_arg
from liken._validators import validate_keep_arg
from liken._validators import validate_spark_args


# API:


class Dedupe(Generic[DF]):
    """Deduplicate a dataframe given a collection of strategies.

    Args:
        df: The dataframe to deduplicate.
        spark_session: optional spark session if initializing with PySpark
            backend.

    Attributes:
        df: Returns the dataframe as currently stored in the `Dedupe` instance's
            strategy manager.
        strats: Returns the strategies as currently stored in the strategy
            manager.

    Raises:
        ValueError: Initialized with PySpark DataFrame but no Spark Session.
    """

    _df: DF
    _executor: Executor[DF]

    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame | spark.DataFrame,
        /,
        *,
        spark_session: SparkSession | None = None,
    ):
        self._df = df

        self._sm = StrategyManager()

        self._executor: LocalExecutor | SparkExecutor
        if isinstance(df, spark.DataFrame):
            spark_session = validate_spark_args(spark_session)
            self._executor = SparkExecutor(spark_session=spark_session)
        else:
            self._executor = LocalExecutor()

    def apply(self, strategy: BaseStrategy | dict | Rules) -> None:
        """Apply a strategy or strategies for deduplication.

        Available for inspection when access with attribute `.strats`. Can be
        repetitively called if using the Sequential API. Else apply once using
        the Dict API or Rules API.

        Args:
            strategy: The strategy or strategies to apply

        Returns:
            None

        Raises:
            InvalidStrategyError: For any invalid strategy or collection of
                strategies

        Example:
            Import and prepate data:

                from liken import Dedupe, exact, tfidf, lsh

                df = ... # get a dataframe

            Sequential API:

                lk = Dedupe(df)
                lk.apply(exact())
                lk.apply(tfidf()) # Ok, apply more than once
                df = lk.drop_duplicates("address")

            Dict API:

                lk = Dedupe(df)
                lk.apply(
                    {
                        "address": (exact(), tfidf()),
                    }
                )
                df = lk.drop_duplicates()

            Dict API:

                from liken.rules import Rules, on

                lk = Dedupe(df)
                lk.apply(
                    Rules(
                        on("address", exact()),
                        on("address", tfidf()),
                    )
                )
                df = lk.drop_duplicates()

        """
        self._sm.apply(strategy)

    def drop_duplicates(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
    ) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
        """Drop duplicates by enacting the applied strategies.

        If no strategies are explicitely provided, will carry out an exact
        deduplication on any number of columns provided in `columns`.
        The `.strats` attribute will return None after calling this function.

        Args:
            columns (str | tuple[str, ...] | None): The attribute(s) of the
                dataframe to deduplicate.
            keep: Accepted as "first" or "last". Whether to keep the first intance
                of a duplicate or the last intance, as found in the DataFrame.

        Returns:
            A deduplicated DataFrame.

        Raises:
            ValueError: Incorrect value to `keep` arg.
            ValueError: Incorrect use of `columns` arg given API used to apply
                strategies.
            ValueError: Incorrect use a single column strategy given multiple
                columns defined, or vice-versa.
        """
        keep = validate_keep_arg(keep)
        columns = validate_columns_arg(columns, self._sm.is_sequential_applied)
        wdf: Frame = wrap(self._df, None)  # canonical id only ever autoincremental for dropping

        # No .apply(), assumes exact deduplication
        if not self._sm.has_applies:
            self._sm.apply(exact())
        strats: StratsDict | Rules = self._sm.get()

        df = self._executor.execute(
            wdf,
            columns=columns,
            strats=strats,
            keep=keep,
            drop_duplicates=True,
            drop_canonical_id=True,
            id=None,
        )

        self._sm.reset()

        return df.unwrap()

    def canonicalize(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
        drop_duplicates: bool = False,
        id: str | None = None,
    ) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
        """Canonicalize by enacting the applied strategies.

        If no strategies are explicitely provided, will carry out an exact
        canonicalization on any number of columns provided in `columns`.
        The `.strats` attribute will return None after calling this function.

        Args:
            columns (str | tuple[str, ...] | None): The attribute(s) of the
                dataframe to deduplicate.
            keep: Accepted as "first" or "last". Whether to keep the first
                intance of a duplicate or the last intance, as found in the
                DataFrame.
            drop_duplicates: Optionally drop duplicates, whilst preserving a
                canonical_id, contrary to `drop_duplicates`.
            id: string label identifying a column in the dataframe that can be
                used to optionally override the values of a default
                canonical_id.

        Returns:
            A canonicalised DataFrame. By default canonicalization is tracked
                in a new `canonical_id` field.

        Raises:
            ValueError: Incorrect value to `keep` arg.
            ValueError: Incorrect use of `columns` arg given API used to apply
                strategies.
            ValueError: Incorrect use a single column strategy given multiple
                columns defined, or vice-versa.
        """
        keep = validate_keep_arg(keep)
        columns = validate_columns_arg(columns, self._sm.is_sequential_applied)
        wdf: Frame = wrap(self._df, id)

        # No .apply(), assumes exact deduplication
        if not self._sm.has_applies:
            self.apply(exact())
        strats: StratsDict | Rules = self._sm.get()

        df: Frame = self._executor.execute(
            wdf,
            columns=columns,
            strats=strats,
            keep=keep,
            drop_duplicates=drop_duplicates,
            drop_canonical_id=False,
            id=id,
        )

        self._sm.reset()

        return df.unwrap()

    @property
    def strats(self) -> str | None:
        """
        Returns the strategies as currently stored in the strategy manager.

        If no strategies are stored, returns None. Otherwise, returns a string
        representation of the strategies containedtuple
        of strategy names or a dictionary mapping attributes to their
        respective strategies.

        Returns:
            The stored strategies, formatted
        """
        return self._sm.pretty_get()
