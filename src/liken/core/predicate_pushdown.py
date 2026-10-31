"""Predicate-step resolution for the LocalExecutor.

A pipeline step containing at least one predicate deduper computes a
visible set: the intersection of all predicate contributions over the
frame. A predicate's contribution is its pair-endpoint set — the rows
appearing in at least one pair its pair generation emits over the
current pool — which is the grouping membership for relational
predicates (`sorted_tokens`, `date_range`) and the per-row match set for
star-shaped ones (`isin`, `str_*`, `num_range`, `isna`). Downstream
dedupers run on the visible set, for grouping and for candidate
selection. Only the step's last deduper builds a union-find.

Contributions are pool-restricted and therefore order-dependent for
relational predicates: a predicate pairs only rows earlier predicates
kept visible. `Pipeline.step` sorts predicates first, so this matches
the historical behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper


if TYPE_CHECKING:
    from collections.abc import Collection

    from liken.collections.pipelines import PipelineStep
    from liken.core.disjoint_set import DisjointSet
    from liken.core.wrapper import DF
    from liken.preprocessors import Preprocessor
    from liken.types import Columns
    from liken.types import MultiComponents
    from liken.types import SingleComponents


# PUBLIC API:


def resolve_visible_pool(step: PipelineStep, wdf: DF) -> set[int] | None:
    """Resolve a step's visible set: the intersection of predicate
    pair-endpoint sets.

    Args:
        step: the pipeline step, a list of (columns, deduper,
            preprocessors) units.
        wdf: the wrapped dataframe.

    Returns:
        `None` when the step has no predicate deduper: the caller falls
        back to the no-predicate branch. Otherwise the visible pool as a
        set of row indices, possibly empty.

    Raises:
        Same errors as the deduper's preprocessing and matching.
    """
    if not any(isinstance(unit[1], PredicateDeduper) for unit in step):
        return None

    pool: set[int] | None = None

    for columns, deduper, preprocessors in step:
        if not isinstance(deduper, PredicateDeduper):
            continue

        match_set = _predicate_match_set(deduper, wdf, columns, preprocessors, pool)

        if pool is None:
            pool = match_set
        else:
            pool &= match_set

        if not pool:
            # An empty intersection cannot grow: every remaining
            # predicate shrinks or keeps it. Short-circuit.
            return pool

    return pool


def _predicate_match_set(
    deduper: PredicateDeduper,
    wdf: DF,
    columns: Columns,
    preprocessors: Collection[Preprocessor],
    pool: set[int] | None,
) -> set[int]:
    """Compute a predicate's pair-endpoint set over the current pool.

    The column is read whole and preprocessed whole, then subset on the
    pool; pair generation runs over the subset and its emitted pair
    endpoints, mapped back through the pool order, are the
    contribution. Pair generation over a subset equals pair generation
    over the full frame restricted to it, so the contribution is stable
    while the pool shrinks.
    """
    array = wdf.get_array(columns)
    processed = deduper.preprocess(array, preprocessors, columns)

    subset = processed.take(pa.array(sorted(pool))) if pool is not None else processed

    endpoints: set[int] = set()
    for i, j in deduper._gen_similarity_pairs(subset):
        endpoints.add(i)
        endpoints.add(j)

    if pool is None:
        return endpoints
    pool_list = sorted(pool)
    return {pool_list[i] for i in endpoints if i < len(pool_list)}


def resolve_predicate_step(
    step: PipelineStep,
    wdf: DF,
) -> SingleComponents | MultiComponents | None:
    """Resolve a pipeline step containing predicate dedupers.

    The visible set is the intersection of all predicate match sets. The
    step's last deduper — predicate or threshold — builds the union-find
    over the visible set; intermediate dedupers build nothing. A row
    failing any predicate is invisible: it is excluded from grouping and
    from candidate selection. An empty visible set yields no grouping.

    Args:
        step: the pipeline step, a list of (columns, deduper,
            preprocessors) units. Predicates are expected sorted first,
            as `Pipeline.step` stores them.
        wdf: the wrapped dataframe.

    Returns:
        `None` when the step has no predicate deduper: the caller falls
        back to the no-predicate branch. Otherwise the components dict in
        the shape the last deduper's union-find produces: original row
        indices mapped to their group's first (smallest) visible index.

    Raises:
        Same errors as the dedupers' preprocessing and matching.
    """
    pool = resolve_visible_pool(step, wdf)

    if pool is None:
        return None

    if not pool:
        # No grouping: every row canonical to itself. `set_frame` still
        # runs — the executor calls the last deduper's canonicalizer,
        # which needs the frame.
        step[-1][1].set_frame(wdf)
        return {}

    pool_list = sorted(pool)

    columns, deduper, preprocessors = step[-1]

    uf, n = _build_union_find(deduper, wdf, columns, preprocessors, pool_list)

    # `build_union_find` subsets on the sorted pool, so union-find roots
    # are subset positions: map each position back to its row index.
    components: SingleComponents = {}
    for pos in range(n):
        components.setdefault(pool_list[uf[pos]], []).append(pool_list[pos])

    return components


def _build_union_find(
    deduper: BaseDeduper,
    wdf: DF,
    columns: Columns,
    preprocessors: Collection[Preprocessor],
    predicate: Collection[int],
) -> tuple[DisjointSet, int]:
    return deduper.set_frame(wdf).build_union_find(columns, preprocessors, predicate=predicate)
