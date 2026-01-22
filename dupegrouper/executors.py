from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, Type, cast, final
from functools import partial

from pyspark.sql import Row, SparkSession

from dupegrouper.constants import CANONICAL_ID
from dupegrouper.dataframe import DF, LocalDF, SparkDF
from dupegrouper.strats_library import BaseStrategy
from dupegrouper.strats_manager import DEFAULT_STRAT_KEY, StratsDict, Rules
from dupegrouper.types import Columns, Keep

if TYPE_CHECKING:
    from dupegrouper.base import Duped


class Executor(Protocol[DF]):
    def execute(
        self,
        df: DF,
        /,
        *,
        columns: Columns | None,
        strats: StratsDict,
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
    ) -> LocalDF:

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
                        components = self._get_components(uf, n)
                        df = call_strat(strat, components)
            else:
                # For inline calls e.g.`.canonicalize("address")`
                for strat in strats[DEFAULT_STRAT_KEY]:
                    uf, n = self._build_uf(strat, df, columns)
                    components = self._get_components(uf, n)
                    df = call_strat(strat, components)

        if isinstance(strats, Rules):
            for stage in strats:
                ufs = []
                for col, strat in stage.and_strats:
                    uf, n = self._build_uf(strat, df, col)
                    ufs.append(uf)
                components = self._get_multi_components(ufs, n)
                df = call_strat(strat, components)

        if drop_canonical_id:
            return df.drop_col(CANONICAL_ID)
        return df

    @staticmethod
    def _build_uf(
        strat: BaseStrategy,
        df: LocalDF,
        columns: Columns,
    ) -> tuple[dict[int, int], int]:
        return strat.set_frame(df).build_union_find(columns)

    @staticmethod
    def _get_components(
        uf: dict[int, int],
        n: int,
    ) -> dict[int, list[int]]:
        components = defaultdict(list)
        for i in range(n):
            components[uf[i]].append(i)
        return components

    @staticmethod
    def _get_multi_components(
        ufs: list[dict[int, int]],
        n: int,
    ) -> dict[tuple[int, ...], list[int]]:
        components = defaultdict(list)
        for i in range(n):
            signature = tuple(uf[i] for uf in ufs)
            components[signature].append(i)
        return components

    @staticmethod
    def _call_strat(
        strat: BaseStrategy,
        components: dict[int | tuple[int, ...], list[int]],
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
        strats: StratsDict,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
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
        from dupegrouper.base import Duped

        # IMPORTANT: Use local variables, no references to Self
        id = self._id
        process_partition = self._process_partition

        rdd = df.mapPartitions(
            lambda partition: process_partition(
                factory=Duped,
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
        factory: Type[Duped[SparkDF]],
        partition: Iterator[Row],
        strats: StratsDict,
        id: str | None,
        columns: Columns | None,
        drop_duplicates: bool,
        keep: Keep = "first",
    ) -> Iterator[Row]:
        """process a spark dataframe partition i.e. a list[Row]

        This function is functionality mapped to a worker node. For clean
        separation from the driver, strats are re-instantiated and the main
        dupegrouper API is executed *per* worker node.

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
        dp = factory(rows, id=id)
        dp.apply(strats)
        dp.canonicalize(
            columns,
            keep=keep,
            drop_duplicates=drop_duplicates,
        )

        return iter(cast(list[Row], dp.df))
