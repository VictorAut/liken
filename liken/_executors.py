"""Strategy executors

`SparkExecutor` simply calls a partition processor where each partition will
then be processed with the `LocalExecutor`
"""

# mypy: disable-error-code="no-redef"

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING
from typing import Protocol
from typing import Type
from typing import TypeAlias
from typing import cast
from typing import final

from networkx.utils.union_find import UnionFind
from pyspark.sql import Row
from pyspark.sql import SparkSession

from liken._constants import CANONICAL_ID
from liken._dataframe import DF
from liken._dataframe import LocalDF
from liken._dataframe import SparkDF
from liken._strats_library import BaseStrategy
from liken._strats_manager import SEQUENTIAL_API_DEFAULT_KEY
from liken._strats_manager import Rules
from liken._strats_manager import StratsDict
from liken._types import Columns
from liken._types import Keep


if TYPE_CHECKING:
    from liken.dedupe import Dedupe


SingleComponents: TypeAlias = dict[int, list[int]]
MultiComponents: TypeAlias = dict[tuple[int, ...], list[int]]


class Executor(Protocol[DF]):
    def execute(
        self,
        df: DF,
        /,
        *,
        columns: Columns | None,
        strats: StratsDict | Rules,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None,
    ) -> DF: ...


@final
class LocalExecutor(Executor):

    def execute(
        self,
        df: LocalDF,
        /,
        *,
        columns: Columns | None,
        strats: StratsDict | Rules,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> LocalDF:
        """Process a local dataframe according to the strategy collection

        Processing is defined according to whether the collections of
        strategies is:
            - Rules: in which case "and" combinations are allowed
            - StratsDict: in which case handles Sequential and Dict API
        """

        del id  # Unused: here for interface symmetry with SparkExecutor

        call_strat = partial(
            self._call_strat,
            drop_duplicates=drop_duplicates,
            keep=keep,
        )

        if isinstance(strats, StratsDict):
            if not columns:
                for col, iter_strats in strats.items():
                    for strat in iter_strats:
                        uf, n = self._build_uf(strat, df, col)
                        components: SingleComponents = self._get_components(uf, n)
                        df = call_strat(strat, components)
            else:
                # For sequence calls e.g.`.canonicalize("address")`
                for strat in strats[SEQUENTIAL_API_DEFAULT_KEY]:
                    uf, n = self._build_uf(strat, df, columns)
                    components: SingleComponents = self._get_components(uf, n)
                    df = call_strat(strat, components)

        if isinstance(strats, Rules):
            for stage in strats:
                ufs = []
                for col, strat in stage.and_strats:
                    uf, n = self._build_uf(strat, df, col)
                    ufs.append(uf)
                components: MultiComponents = self._get_multi_components(ufs, n)
                df = call_strat(strat, components)

        if drop_canonical_id:
            return df.drop_col(CANONICAL_ID)
        return df

    @staticmethod
    def _build_uf(
        strat: BaseStrategy,
        df: LocalDF,
        columns: Columns,
    ) -> tuple[UnionFind[int], int]:
        return strat.set_frame(df).build_union_find(columns)

    @staticmethod
    def _get_components(
        uf: UnionFind[int],
        n: int,
    ) -> SingleComponents:
        components = defaultdict(list)
        for i in range(n):
            components[uf[i]].append(i)
        return components

    @staticmethod
    def _get_multi_components(
        ufs: list[UnionFind[int]],
        n: int,
    ) -> MultiComponents:
        components = defaultdict(list)
        for i in range(n):
            signature = tuple(uf[i] for uf in ufs)
            components[signature].append(i)
        return components

    @staticmethod
    def _call_strat(
        strat: BaseStrategy,
        components: SingleComponents | MultiComponents,
        drop_duplicates: bool,
        keep: Keep,
    ) -> LocalDF:
        return strat.canonicalizer(
            components=components,
            drop_duplicates=drop_duplicates,
            keep=keep,
        )


@final
class SparkExecutor(Executor):

    def __init__(self, spark_session: SparkSession, id: str | None = None):
        self._spark_session = spark_session
        self._id = id

    def execute(
        self,
        df: SparkDF,
        /,
        *,
        columns: Columns | None,
        strats: StratsDict | Rules,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> SparkDF:
        """Spark specific deduplication helper

        Maps dataframe partitions to be processed via the RDD API yielding low-
        level list[Rows], which are then post-processed back to a dataframe.

        Args:
            columns: The attribute to deduplicate.
            strats: the collection of strats
        Retuns:
            Instance's _df attribute is updated
        """

        # import in worker node
        from liken.dedupe import Dedupe

        # IMPORTANT: Use local variables, no references to Self
        process_partition = self._process_partition

        rdd = df.mapPartitions(
            lambda partition: process_partition(
                factory=Dedupe,
                partition=partition,
                strats=strats,
                id=id,
                columns=columns,
                drop_duplicates=drop_duplicates,
                keep=keep,
            )
        )

        schema = df._schema

        df = SparkDF(self._spark_session.createDataFrame(rdd, schema=schema), is_init=False)

        if drop_canonical_id:
            return df.drop_col(CANONICAL_ID)
        return df

    @staticmethod
    def _process_partition(
        *,
        factory: Type[Dedupe[SparkDF]],
        partition: Iterator[Row],
        strats: StratsDict | Rules,
        id: str | None,
        columns: Columns | None,
        drop_duplicates: bool,
        keep: Keep = "first",
    ) -> Iterator[Row]:
        """process a spark dataframe partition i.e. a list[Row]

        This function is functionality mapped to a worker node. For clean
        separation from the driver, strats are re-instantiated and the main
        liken API is executed *per* worker node.

        Args:
            paritition_iter: a partition
            strats: the collection of strats
            id: the unique identified of the dataset a.k.a "business key"
            columns: the attribute on which to deduplicate

        Returns:
            A list[Row], deduplicated
        """
        # handle empty partitions
        rows: list[Row] = list(partition)
        if not rows:
            return iter([])

        # Core API reused per partition, per worker node
        lk = factory(rows)  # TODO: typing needs to be thought about here for `Dedupe` given `Rows`
        lk.apply(strats)  # type: ignore
        df = lk.canonicalize(
            columns,
            keep=keep,
            drop_duplicates=drop_duplicates,
            id=id,
        )

        return iter(cast(list[Row], df))
