"""liken main public API"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Self

from liken.collections.base import CollectionsManager
from liken.core.dispatcher import get_backend
from liken.core.dispatcher import wrap
from liken.core.executor import Executor
from liken.core.executor import LocalExecutor
from liken.dedupers.exact import exact
from liken.explore import DEFAULT_EXPLORE_THRESHOLDS
from liken.explore import run_explore
from liken.validators import validate_columns_arg
from liken.validators import validate_explore_columns_arg
from liken.validators import validate_frac_arg
from liken.validators import validate_keep_arg
from liken.validators import validate_spark_arg
from liken.validators import validate_thresholds_arg


if TYPE_CHECKING:
    from collections.abc import Hashable

    from pyspark.sql import Row
    from pyspark.sql import SparkSession

    from liken.collections.dict import DeduplicationDict
    from liken.collections.pipelines import Pipeline
    from liken.core.backend import Backend
    from liken.core.deduper import BaseDeduper
    from liken.core.wrapper import DF
    from liken.types import Columns
    from liken.types import InternalDataFrame
    from liken.types import Keep
    from liken.types import UserDataFrame


class Dedupe:
    """Deduplicate a dataframe given a collection of dedupers.

    Apply a deduper as a method call

    Args:
        df: The dataframe to deduplicate.
        spark_session: optional spark session if initializing with PySpark
            backend.

    Raises:
        ValueError: Initialized with PySpark DataFrame but no Spark Session.

    Examples:

        import liken as lk

        lk.dedupe(df).apply(exact()).drop_duplicates()
    """

    _executor: Executor

    def __init__(
        self,
        df: UserDataFrame,
        /,
        *,
        spark_session: SparkSession | None = None,
    ):
        self._df: InternalDataFrame = df

        self._collection = CollectionsManager()

        backend: Backend = get_backend(self._df)

        if backend.name == "pyspark":
            spark_session = validate_spark_arg(spark_session)

        self._executor: Executor = backend.executor(spark_session=spark_session)

        self.has_been_canonicalized: bool = False

    @classmethod
    def _from_rows(
        cls,
        rows: list[Row],
    ) -> Dedupe:
        """bypass initialization and initialize explicitly with no validation.

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
            Self

        Raises:
            InvalidDeduperError: For any invalid deduper or collection of
                dedupers

        Example:
            Import and prepare data:

                import liken as lk

            Simple API:

                lk.dedupe(df).apply(lk.exact())

            Dict API:

                lk.dedupe(df).apply({"address": (exact(), tfidf())})

            Pipeline API:

                lk.dedupe(df).apply(
                    lk.pipeline()
                    .step(lk.col("address").exact())
                    .step(lk.col("address").tfidf())
                )

        """
        self._collection.apply(deduper)
        return self

    def explore(
        self,
        columns: list[str] | dict[str, BaseDeduper],
        *,
        thresholds: list[float] | None = None,
        frac: float = 1.0,
    ) -> UserDataFrame:
        """Profile the potential duplicate rate of one or more columns.

        Inspired by pandas `DataFrame.describe`, `explore` is an exploratory
        tool to understand whether, and how aggressively, data deduplicates —
        without committing to a deduplication. For each column it reports the
        duplicate rate (the fraction of rows that are redundant duplicates, i.e.
        that would be removed by `drop_duplicates`) for exact matching and for
        similarity matching across a sweep of thresholds.

        By default each column is analysed with the `fuzzy` deduper. Pass a dict
        mapping a column to a single-column similarity deduper to use a
        different deduper for that column; it is swept across the same
        `thresholds`.

        As deduplication scales at approximately O(n^2), use the `frac` arg to
        analyse a random sample of the data for a faster, approximate result.

        Note:
            Only supported for the pandas, polars, modin and pyarrow backends.

        Args:
            columns (list[str] | dict[str, BaseDeduper]): The column labels to
                analyse with the default `fuzzy` deduper, or a dict mapping a
                column label to a single-column similarity (threshold) deduper
                to use for that column.
            thresholds: The similarity thresholds to sweep, each a float in the
                range (0, 1). Defaults to [0.5, 0.75, 0.9, 0.95, 0.99].
            frac: The fraction of rows to randomly sample before analysis, a
                float in the range (0, 1]. Defaults to 1.0 (no sampling).

        Returns:
            A dataframe, in the same backend as the input, of duplicate rates.

        Raises:
            ValueError: Unsupported backend, invalid `frac`, invalid
                `thresholds`, invalid `columns`, or a column not in the
                dataframe.
        """
        validate_frac_arg(frac)
        thresholds = validate_thresholds_arg(thresholds if thresholds is not None else DEFAULT_EXPLORE_THRESHOLDS)
        validate_explore_columns_arg(columns)

        return run_explore(self._df, columns, thresholds, frac)

    def drop_duplicates(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
    ) -> UserDataFrame:
        """Drop duplicates by enacting the applied dedupers.

        If no dedupers are explicitly provided, will carry out an exact
        deduplication on any number of columns provided in `columns`.

        Args:
            columns (str | tuple[str, ...] | None): The attribute(s) of the
                dataframe to deduplicate.
            keep: Accepted as "first" or "last". Whether to keep the first instance
                of a duplicate or the last instance, as found in the DataFrame.

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
        wdf: DF = wrap(self._df, None)  # canonical id only ever autoincremental for dropping

        # No .apply(), assumes exact deduplication
        if not self._collection.has_applies:
            self._collection.apply(exact())
        dedupers: DeduplicationDict | Pipeline = self._collection.get()

        self._df: InternalDataFrame = self._executor.execute(
            wdf,
            columns=columns,
            dedupers=dedupers,
            keep=keep,
            drop_duplicates=True,
            drop_canonical_id=True,
            id=None,
        ).unwrap()

        self._collection.reset()

        return self._df

    def canonicalize(
        self,
        columns: Columns | None = None,
        *,
        keep: Keep = "first",
        drop_duplicates: bool = False,
        id: str | None = None,
    ) -> Self:
        """Canonicalize by enacting the applied dedupers.

        If no dedupers are explicitly provided, will carry out an exact
        canonicalization on any number of columns provided in `columns`.

        Warning:
            Leaving `id` to it's default `None` value forces collection to
            driver node when using `Ray` Datasets and `Dask` DataFrames, which
            is not recommended. Use the dataset's unique identifier with the `id`
            arg, instead.

        Args:
            columns (str | tuple[str, ...] | None): The attribute(s) of the
                dataframe to deduplicate.
            keep: Accepted as "first" or "last". Whether to keep the first
                instance of a duplicate or the last instance, as found in the
                DataFrame.
            drop_duplicates: Optionally drop duplicates, whilst preserving a
                canonical_id, contrary to `drop_duplicates`.
            id: string label identifying a column in the dataframe that can be
                used to optionally override the values of a default
                canonical_id.

        Returns:
            Self. Access the dataframe with `.collect`, or numbers of repeated
                canonicals ids with `.canonicals`, or synthetic records with
                `.synthesize`.

        Raises:
            ValueError: Incorrect value to `keep` arg.
            ValueError: Incorrect use of `columns` arg given API used to apply
                dedupers.
            ValueError: Incorrect use of a single column deduper given multiple
                columns defined, or vice-versa.
        """
        keep: Keep = validate_keep_arg(keep)
        columns: Columns | None = validate_columns_arg(columns, self._collection.is_sequential_applied)
        wdf: DF = wrap(self._df, id)

        # No .apply(), assumes exact deduplication
        if not self._collection.has_applies:
            self.apply(exact())
        dedupers: DeduplicationDict | Pipeline = self._collection.get()

        self._df: InternalDataFrame = self._executor.execute(
            wdf,
            columns=columns,
            dedupers=dedupers,
            keep=keep,
            drop_duplicates=drop_duplicates,
            drop_canonical_id=False,
            id=id,
        ).unwrap()

        self._collection.reset()

        self.has_been_canonicalized: bool = True

        return self

    def canonicals(self, n: int = 2) -> dict[Hashable, int]:
        """Returns a dictionary of canonical ids that have `n` or more records.

        Only allows n>=2. Only valid for deduplication with canonicalization.

        Args:
            n: the number of records per canonical id, defaulted at 2

        Warning:
            For PySpark dataframes, Dask dataframes and Ray datasets, this
            function forces the collection of data to the driver node.
            Additionally, this function only supports usage with PySpark `v4`
            and up.

        Returns:
            A dictionary of canonical ids, where values are counts.

        Raises:
            ValueError: Incorrect `n`
            RuntimeError: When called before `canonicalize`
        """

        if n < 2:
            raise ValueError("n must be >= 2")

        if not self.has_been_canonicalized:
            raise RuntimeError("No canonical_id counts found. Run `.canonicalize()` first.")

        wdf: DF = wrap(self._df, id=None)

        canonical_array: list[str | int] = wdf.get_canonical().to_pylist()

        counts: dict[Hashable, int] = {}
        for cid in canonical_array:
            counts[cid] = counts.get(cid, 0) + 1

        return {cid: count for cid, count in counts.items() if count >= n}

    def synthesize(self) -> UserDataFrame:
        """Synthesizes a record combining the first instance of non null values
        of all records associated to a canonical id.

        The resulting "golden" record essentially contains coalesced values of
        all attributes, for the given set of associated records.

        In the case of canonical records that only have one associated record,
        they are returned as-is.

        Info:
            A future version of `.synthesize` will allow for picking a chosen
            value given a set of records. The current implementation is limited
            to the `first` instance, but `last` will also be supported as well
            as `min` and `max` for numerical data.

        Warning:
            For PySpark dataframes, Dask dataframes and Ray datasets, this
            function forces the collection of data to the driver node.
            Additionally, this function only supports usage with PySpark `v4`
            and up.

        Returns:
            A dataframe of synthesized records.
        """

        wdf: DF = wrap(self._df, id=None)

        return wdf.synthesize_record()

    def collect(self) -> UserDataFrame:
        """Collect canonicalization results and returns the dataframe."""
        return self._df

    def explain(self) -> str | None:
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
