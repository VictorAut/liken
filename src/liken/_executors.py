"""Deduplication collectionexecutors.

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
from typing import TypeVar
from typing import cast
from typing import final

import pandas as pd
import ray
from networkx.utils.union_find import UnionFind
from pyspark.sql import Row
from pyspark.sql import SparkSession

from liken._collections import SEQUENTIAL_API_DEFAULT_KEY
from liken._collections import DeduplicationDict
from liken._collections import Pipeline
from liken._constants import CANONICAL_ID
from liken._dataframe import DaskDF
from liken._dataframe import Frame
from liken._dataframe import LocalDF
from liken._dataframe import RayDF
from liken._dataframe import SparkDF
from liken._dedupers import BaseDeduper
from liken._dedupers import PredicateDedupers
from liken._preprocessors import Preprocessor
from liken._types import Columns
from liken._types import Keep


if TYPE_CHECKING:
    from liken.liken import Dedupe


# TYPES:


SingleComponents: TypeAlias = dict[int, list[int]]
MultiComponents: TypeAlias = dict[tuple[int, ...], list[int]]
F = TypeVar("F", bound=Frame)


# EXECUTORS:


class Executor(Protocol[F]):
    def execute(
        self,
        df: F,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None,
    ) -> F: ...


@final
class LocalExecutor(Executor):
    def execute(
        self,
        df: LocalDF,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> LocalDF:
        """Process a local dataframe according to the deduplication collection

        Processing is defined according to whether the collections of
        dedupers is a:
            - Pipeline: in which case "and" combination steps are allowed
            - DeduplicationDict: in which case handles Sequential and Dict API

        For Pipeline, predication is implemented if an and combination contains at
        least one Predicate Deduper. In that case the predicate dedupers are proxy
        WHERE filters that propagate a set of dataframe indice positions to the
        next deduper (most likely a threshold deduper, but optionally predicate
        also.)
        """

        del id  # Unused: here for interface symmetry with SparkExecutor

        call_deduper = partial(
            self._call_deduper,
            drop_duplicates=drop_duplicates,
            keep=keep,
        )

        if isinstance(dedupers, DeduplicationDict):
            if not columns:
                for col, iter_dedupers in dedupers.items():
                    for deduper in iter_dedupers:
                        uf, n = self._build_uf(deduper, df, col)
                        components: SingleComponents = self._get_components(uf, n)
                        df = call_deduper(deduper, components)
            else:
                # For sequential API calls e.g.`.canonicalize("address")`
                for deduper in dedupers[SEQUENTIAL_API_DEFAULT_KEY]:
                    uf, n = self._build_uf(deduper, df, columns)
                    components: SingleComponents = self._get_components(uf, n)
                    df = call_deduper(deduper, components)

        if isinstance(dedupers, Pipeline):
            for step in dedupers.steps:
                any_predicate: bool = dedupers._has_any_predicate(step)

                # predication only if at least one predicate deduper
                if any_predicate:
                    indices = set()

                    for col, deduper, preprocessor in step:
                        uf, n = self._build_uf(
                            deduper, df, col, preprocessor, predicate=indices
                        )

                        components = defaultdict(list)
                        idx: list = sorted(indices)
                        for i in range(n):
                            if not indices:
                                components[uf[i]].append(i)
                            else:
                                components[idx[uf[i]]].append(idx[i])

                        if isinstance(deduper, PredicateDedupers):
                            for c in components.values():
                                if len(c) > 1:
                                    indices = indices.union(set(c))

                else:
                    ufs = []

                    for col, deduper, preprocessor in step:
                        uf, n = self._build_uf(deduper, df, col, preprocessor)
                        ufs.append(uf)
                    components: MultiComponents = self._get_multi_components(ufs, n)

                df = call_deduper(deduper, components)

        if drop_canonical_id:
            return df.drop_col(CANONICAL_ID)
        return df

    @staticmethod
    def _build_uf(
        deduper: BaseDeduper,
        df: LocalDF,
        columns: Columns,
        preprocessors: list[Preprocessor] = [],
        predicate: set = set(),
    ) -> tuple[UnionFind[int], int]:
        return deduper.set_frame(df).build_union_find(
            columns, preprocessors, predicate=predicate
        )

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
    def _call_deduper(
        deduper: BaseDeduper,
        components: SingleComponents | MultiComponents,
        drop_duplicates: bool,
        keep: Keep,
    ) -> LocalDF:
        return deduper.canonicalizer(
            components=components,
            drop_duplicates=drop_duplicates,
            keep=keep,
        )


@final
class SparkExecutor(Executor):
    def __init__(self, spark_session: SparkSession):
        self._spark_session = spark_session

    def execute(
        self,
        df: SparkDF,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> SparkDF:
        """Maps dataframe partitions to be processed via the RDD API yielding
        low-level list[Rows], which are then post-processed back to a dataframe.
        """

        # import in worker node
        from liken.liken import Dedupe

        # IMPORTANT: Use local variables, no references to Self
        # Allows for serialization via Py4J
        process_partition = self._process_partition

        rdd = df.mapPartitions(
            lambda partition: process_partition(
                factory=Dedupe,
                partition=partition,
                dedupers=dedupers,
                id=id,
                columns=columns,
                drop_duplicates=drop_duplicates,
                keep=keep,
            )
        )

        schema = df._schema

        df = SparkDF(
            self._spark_session.createDataFrame(rdd, schema=schema), is_init=False
        )

        if drop_canonical_id:
            return df.drop_col(CANONICAL_ID)
        return df

    @staticmethod
    def _process_partition(
        *,
        factory: Type[Dedupe],
        partition: Iterator[Row],
        dedupers: DeduplicationDict | Pipeline,
        id: str | None,
        columns: Columns | None,
        drop_duplicates: bool,
        keep: Keep = "first",
    ) -> Iterator[Row]:
        """process a spark dataframe partition i.e. a list[Row]

        This function is functionality mapped to a worker node. For clean
        separation from the driver, dedupers are re-instantiated and the main
        liken API is executed *per* worker node.

        Args:
            paritition_iter: a partition
            dedupers: the collection of dedupers
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
        df = (
            factory._from_rows(rows)
            .apply(dedupers)
            .canonicalize(
                columns,
                keep=keep,
                drop_duplicates=drop_duplicates,
                id=id,
            )
            .collect()
        )

        return iter(cast(list[Row], df))


class RayExecutor(Executor):
    def __init__(self):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def execute(
        self,
        df: RayDF,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> RayDF:
        """Maps dataframe partitions to be processed by Ray enforced to use
        pandas for each partition.
        """

        from liken.liken import Dedupe

        def _process_batch(batch: pd.DataFrame) -> pd.DataFrame:
            return (
                Dedupe(batch)
                .apply(dedupers)
                .canonicalize(
                    columns,
                    keep=keep,
                    drop_duplicates=drop_duplicates,
                    id=id,
                )
                .collect()
            )

        # IMPORTANT: "pandas" batch
        df = RayDF(df._df.map_batches(_process_batch, batch_format="pandas"))

        if drop_canonical_id:
            return df.drop_col("canonical_id")
        return df


@final
class DaskExecutor(Executor):
    def execute(
        self,
        df: DaskDF,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> DaskDF:
        """Maps dataframe partitions to be processed by Dask, natively using
        pandas for each partition.
        """
        
        meta = df._new_meta(df._df, id)

        if drop_canonical_id:
            meta = meta.drop(columns=[CANONICAL_ID])

        process_partition = self._process_partition

        df = DaskDF(
            df._df.map_partitions(
                process_partition,
                dedupers=dedupers,
                columns=columns,
                keep=keep,
                drop_duplicates=drop_duplicates,
                drop_canonical_id=drop_canonical_id,
                id=id,
                meta=meta,
            ),
            id=id,
            preserve_schema=True
        )

        # if drop_canonical_id:
        #     return df.drop_col(CANONICAL_ID)
        # return df
        
        return df

    @staticmethod
    def _process_partition(
        df: pd.DataFrame,
        dedupers: DeduplicationDict | Pipeline,
        columns: Columns | None,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        id: str | None = None,
    ) -> pd.DataFrame:
        from liken.liken import Dedupe

        df = (
            Dedupe(df)
            .apply(dedupers)
            .canonicalize(
                columns,
                keep=keep,
                drop_duplicates=drop_duplicates,
                id=id,
            )
            .collect()
        )

        print("columns:", df.columns)

        if drop_canonical_id:
            if CANONICAL_ID in df.columns:
                df = df.drop(columns=[CANONICAL_ID])
        else:
            # Ensure it ALWAYS exists when expected
            if CANONICAL_ID not in df.columns:
                df[CANONICAL_ID] = pd.Series(dtype="int64")
        return df


