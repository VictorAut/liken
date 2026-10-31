"""Deduplication collection executors.

`PysparkExecutor` simply calls a partition processor where each partition will
then be processed with the `LocalExecutor`
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar

from liken.collections.base import Pipeline
from liken.collections.dict import DeduplicationDict
from liken.constants import SEQUENTIAL_API_DEFAULT_KEY
from liken.core.predicate_pushdown import resolve_predicate_step
from liken.core.wrapper import DF


if TYPE_CHECKING:
    from collections.abc import Collection

    from liken.core.deduper import BaseDeduper
    from liken.core.disjoint_set import DisjointSet
    from liken.preprocessors import Preprocessor
    from liken.types import Columns
    from liken.types import Keep
    from liken.types import MultiComponents
    from liken.types import SingleComponents


# TYPES:


D = TypeVar("D", bound=DF)


# EXECUTORS:


class Executor(Protocol[D]):
    def execute(
        self,
        df: D,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        source_id: str | None,
        canonical_id: str,
    ) -> D: ...


class LocalExecutor(Executor):
    def execute(
        self,
        df: DF,
        /,
        *,
        columns: Columns | None,
        dedupers: DeduplicationDict | Pipeline,
        keep: Keep,
        drop_duplicates: bool,
        drop_canonical_id: bool,
        source_id: str | None = None,
        canonical_id: str,
    ) -> DF:
        """Process a local dataframe according to the deduplication collection

        Processing is defined according to whether the collections of
        dedupers is a:
            - Pipeline: in which case "and" combination steps are allowed
            - DeduplicationDict: in which case handles Sequential and Dict API

        For Pipeline, predicate evaluation applies when an "and" combination contains at
        least one Predicate Deduper. In that case the predicate dedupers are proxy
        WHERE filters that propagate a set of dataframe indice positions to the
        next deduper (most likely a threshold deduper, but optionally predicate
        also.)
        """

        del source_id  # Unused: here for interface symmetry with PysparkExecutor

        call_deduper = partial(
            self._call_deduper,
            drop_duplicates=drop_duplicates,
            keep=keep,
            canonical_id=canonical_id,
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
                # predicate steps resolve a visible set: the intersection
                # of the predicate match sets; the step's last deduper
                # builds the union-find over it. `None` means no predicate
                # deduper: the signature branch runs on the full frame.
                components: SingleComponents | MultiComponents = resolve_predicate_step(step, df)

                if components is None:
                    ufs = []

                    for col, deduper, preprocessor in step:
                        uf, n = self._build_uf(deduper, df, col, preprocessor)
                        ufs.append(uf)
                    components = self._get_multi_components(ufs, n)

                df = call_deduper(step[-1][1], components)

        if drop_canonical_id:
            return df.drop_col(canonical_id)
        return df

    @staticmethod
    def _build_uf(
        deduper: BaseDeduper,
        df: DF,
        columns: Columns,
        preprocessors: Collection[Preprocessor] = (),
        predicate: Collection[int] = (),
    ) -> tuple[DisjointSet, int]:
        return deduper.set_frame(df).build_union_find(columns, preprocessors, predicate=predicate)

    @staticmethod
    def _get_components(
        uf: DisjointSet,
        n: int,
    ) -> SingleComponents:
        components = defaultdict(list)
        for i in range(n):
            components[uf[i]].append(i)
        return components

    @staticmethod
    def _get_multi_components(
        ufs: list[DisjointSet],
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
        canonical_id: str,
    ) -> DF:
        return deduper.canonicalizer(
            components=components,
            drop_duplicates=drop_duplicates,
            keep=keep,
            canonical_id=canonical_id,
        )
