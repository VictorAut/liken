"""liken main public API"""

from __future__ import annotations

from typing import Self

import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.sql import SparkSession

from liken._collections import CollectionsManager
from liken._collections import DeduplicationDict
from liken._dataframe import Frame
from liken._dataframe import wrap
from liken._dedupers import BaseDeduper
from liken._dedupers import exact
from liken._executors import Executor
from liken._executors import LocalExecutor
from liken._executors import SparkExecutor
from liken._types import Columns
from liken._types import DataFrameLike
from liken._types import Keep
from liken._types import UserDataFrame
from liken._validators import validate_columns_arg
from liken._validators import validate_df_arg
from liken._validators import validate_keep_arg
from liken._validators import validate_spark_arg
from liken.rules import Pipeline


class Dedupe:
    """Deduplicate a dataframe given a collection of dedupers.

    Apply a deduper as a method call

    Args:
        df: The dataframe to deduplicate.
        spark_session: optional spark session if initializing with PySpark
            backend.

    Returns:
        An Dedupe object with which to "apply" a deduper

    Raises:
        ValueError: Initialized with PySpark DataFrame but no Spark Session.

    Examples:

        import liken as lk

        lk.dedupe(df).apply(exact()).drop_duplicates().collect()
    """

    _executor: Executor

    def __init__(
        self,
        df: UserDataFrame,
        /,
        *,
        spark_session: SparkSession | None = None,
    ):
        self._df: DataFrameLike = validate_df_arg(df)

        self._collection = CollectionsManager()

        if isinstance(df, spark.DataFrame):
            spark_session = validate_spark_arg(spark_session)
            self._executor = SparkExecutor(spark_session=spark_session)
        else:
            self._executor = LocalExecutor()

    @classmethod
    def _from_rows(
        cls,
        rows: list[spark.Row],
    ) -> Dedupe:
        """bypass initialization and initialize explicitely with no validation.

        Use as internal constructor with spark `Rows`.
        """
        self = cls.__new__(cls)
        self._df = rows
        self._collection = CollectionsManager()
        self._executor = LocalExecutor()
        return self

    def apply(self, deduper: BaseDeduper | dict | Pipeline) -> Self:
        """Apply a deduper or dedupers for deduplication.

        Available for inspection when accessed with `.explain()`. Can be
        repetitively called if using the Sequential API. Else apply once using
        the Dict API or Pipeline API.

        Args:
            deduper: The deduper or dedupers to apply

        Returns:
            None

        Raises:
            InvalidDeduperError: For any invalid deduper or collection of
                dedupers

        Example:
            Import and prepate data:

                import liken as lk

            Simple API:

                lk.dedupe(df).apply(lk.exact())

            Dict API:

                lk.dedupe(df).apply({"address": (exact(), tfidf())}

            Pipeline API:

                lk.dedupe(df).apply(
                    lk.rules.pipeline()
                    .step(lk.rules.on("address").exact())
                    .step(lk.rules.on("address").tfidf())
                )

        """
        self._collection.apply(deduper)
        return self

    def drop_duplicates(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
    ) -> Self:
        """Drop duplicates by enacting the applied dedupers.

        If no dedupers are explicitely provided, will carry out an exact
        deduplication on any number of columns provided in `columns`.

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
                dedupers.
            ValueError: Incorrect use a single column deduper given multiple
                columns defined, or vice-versa.
        """
        keep: Keep = validate_keep_arg(keep)
        columns: Columns | None = validate_columns_arg(columns, self._collection.is_sequential_applied)
        wdf: Frame = wrap(self._df, None)  # canonical id only ever autoincremental for dropping

        # No .apply(), assumes exact deduplication
        if not self._collection.has_applies:
            self._collection.apply(exact())
        dedupers: DeduplicationDict | Pipeline = self._collection.get()

        self._df: DataFrameLike = self._executor.execute(
            wdf,
            columns=columns,
            dedupers=dedupers,
            keep=keep,
            drop_duplicates=True,
            drop_canonical_id=True,
            id=None,
        ).unwrap()

        self._collection.reset()

        return self

    def canonicalize(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
        drop_duplicates: bool = False,
        id: str | None = None,
    ) -> Self:
        """Canonicalize by enacting the applied dedupers.

        If no dedupers are explicitely provided, will carry out an exact
        canonicalization on any number of columns provided in `columns`.

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
                dedupers.
            ValueError: Incorrect use of a single column deduper given multiple
                columns defined, or vice-versa.
        """
        keep: Keep = validate_keep_arg(keep)
        columns: Columns | None = validate_columns_arg(columns, self._collection.is_sequential_applied)
        wdf: Frame = wrap(self._df, id)

        # No .apply(), assumes exact deduplication
        if not self._collection.has_applies:
            self.apply(exact())
        dedupers: DeduplicationDict | Pipeline = self._collection.get()

        self._df: DataFrameLike = self._executor.execute(
            wdf,
            columns=columns,
            dedupers=dedupers,
            keep=keep,
            drop_duplicates=drop_duplicates,
            drop_canonical_id=False,
            id=id,
        ).unwrap()

        self._collection.reset()

        return self

    def collect(self) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
        """Collect results and return the dataframe."""
        return self._df

    def explain(self):
        """
        Returns the dedupers as currently stored in the collections manager.

        If no dedupers are stored, returns None. Otherwise, returns a string
        representation of the dedupers collection

        Returns:
            The stored dedupers, formatted

        Examples:

            >>> pipeline = {"address": (lk.exact(), lk.tfidf()), "email": lk.fuzzy()}

            >>> print(lk.dedupe(df).apply(pipeline).explain())

            {
                'address': (
                    exact(),
                    tfidf(threshold=0.95, ngram=3, topn=2),
                    ),
                'email': (
                    fuzzy(threshold=0.95, scorer='simple_ratio'),
                    ),
            }
        """
        return self._collection.pretty_get()


# API:


def dedupe(df: UserDataFrame, /, *, spark_session: SparkSession | None = None) -> Dedupe:
    """Convenience function for `Dedupe` entrypoint."""
    return Dedupe(df, spark_session=spark_session)
