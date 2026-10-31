"""Unit tests for the predicate-pushdown module (WI071).

Pins the visible-set semantics: predicates are pre-filters, the visible
set is the intersection of predicate match sets, and downstream dedupers
run on the visible set.
"""

from unittest.mock import Mock

import pyarrow as pa

from liken.collections.pipelines import PipelineUnit
from liken.core.predicate_pushdown import resolve_visible_pool
from liken.dedupers.isin import IsIn
from liken.dedupers.isna import IsNA
from liken.dedupers.str_startswith import StrStartsWith


def make_df(columns: dict) -> Mock:
    """Minimal DF wrapper stub: get_array dispatches on columns."""
    df = Mock()
    df.distributed = False

    def get_array(cols):
        if isinstance(cols, str):
            return pa.array(columns[cols])
        return pa.table({col: pa.array(values) for col, values in zip(cols, columns.values(), strict=True)})

    df.get_array.side_effect = get_array
    return df


def unit(columns, deduper, preprocessors=()):
    return PipelineUnit(columns, deduper, list(preprocessors))


# Step resolution: components from the last deduper  #


class TrackingDeduper:
    """Stub threshold deduper recording build_union_find calls."""

    def __init__(self, pairs=()):
        self._pairs = pairs
        self.calls = 0
        self.seen_predicates: list[object] = []

    def set_frame(self, wdf):
        return self

    def validate(self, columns):
        pass

    def build_union_find(self, columns, preprocessors, predicate=()):
        from liken.core.disjoint_set import DisjointSet

        self.calls += 1
        self.seen_predicates.append(predicate)

        n = len(columns)
        uf = DisjointSet(n)
        for i, j in self._pairs:
            uf.union(i, j)
        return uf, n

    def canonicalizer(self, *, components, drop_duplicates, keep, canonical_id):
        del components, drop_duplicates, keep, canonical_id
        return


class TestResolvePredicateStep:
    def test_no_predicates_returns_none(self):
        """A step without predicate dedupers resolves to `None`: the caller
        uses the no-predicate signature branch."""
        from liken.core.predicate_pushdown import resolve_predicate_step
        from liken.dedupers.exact import Exact

        df = make_df({"col": ["a", "b", "a"]})
        step = [unit("col", Exact())]

        assert resolve_predicate_step(step, df) is None

    def test_single_predicate_then_threshold_groups_within_pool(self):
        """A row failing any predicate is invisible to the threshold
        deduper: excluded from grouping and from candidate selection."""
        from liken.core.predicate_pushdown import resolve_predicate_step
        from liken.dedupers.exact import Exact

        df = make_df({"c1": ["x", "x", "y"], "c2": ["A", "B", "A"]})
        step = [unit("c1", IsIn(["x"])), unit("c2", Exact())]

        components = resolve_predicate_step(step, df)

        # rows 0, 1 match the predicate; exact on c2 keeps them apart
        assert components == {0: [0], 1: [1]}

    def test_multi_predicate_step_groups_nothing_when_row_fails_one(self):
        """Spec acceptance case: `[Isin("col1", ["x"]), Isin("col2", ["A"])]`
        with rows `(x, A)`, `(x, B)` groups nothing. Old behaviour: the
        pool grew by union of predicate components, so the last deduper
        could group `(x, B)` into the pool. The components dict is empty
        here: the second predicate, restricted to the rows the first kept
        visible, sees a single `A` and contributes nothing, emptying the
        pool. Canonicalisation is unaffected by the dict shape — both
        `{}` and singleton components leave every row its own canonical."""
        from liken.core.predicate_pushdown import resolve_predicate_step

        df = make_df({"col1": ["x", "x"], "col2": ["A", "B"]})
        step = [unit("col1", IsIn(["x"])), unit("col2", IsIn(["A"]))]

        components = resolve_predicate_step(step, df)

        assert components == {}

    def test_empty_visible_set_yields_no_grouping(self):
        """An empty visible set zeroes the step: empty components, every
        row canonical to itself via the canonicalizer's default. The
        threshold deduper builds no union-find either."""
        from liken.core.predicate_pushdown import resolve_predicate_step

        tracker = TrackingDeduper(pairs=((0, 1),))
        # Exact("col2") alone would group nothing here; a grouping frame
        # keeps the assertion meaningful if the fixture changes
        df = make_df({"col1": ["a", "a"], "col2": ["x", "y"]})
        step = [unit("col1", StrStartsWith("zzz")), unit("col2", tracker)]

        components = resolve_predicate_step(step, df)

        assert components == {}
        assert tracker.calls == 0

    def test_intermediate_threshold_deduper_builds_no_union_find(self):
        """Only the last deduper in a predicate step builds a union-find.
        Old behaviour: every deduper built one and intermediates'
        components were discarded."""
        from liken.core.predicate_pushdown import resolve_predicate_step

        tracker = TrackingDeduper(pairs=((0, 1),))

        df = make_df({"c1": ["x", "x", "y"], "c2": ["A", "A", "A"]})
        # predicates sort first in a real step; emulate that order
        step = [unit("c1", IsIn(["x"])), unit("c2", tracker)]

        components = resolve_predicate_step(step, df)

        assert tracker.calls == 1
        # the threshold deduper ran on the visible pool only
        assert tracker.seen_predicates[0] == [0, 1]
        assert components == {0: [0, 1]}

    def test_all_predicate_step_star_groups_the_pool(self):
        """An all-predicate step: the last predicate builds the union-find
        and star-groups the visible rows."""
        from liken.core.predicate_pushdown import resolve_predicate_step

        df = make_df({"c1": ["x", "x", "y"], "c2": ["A", "A", "B"]})
        step = [unit("c1", IsIn(["x"])), unit("c2", IsIn(["A"]))]

        components = resolve_predicate_step(step, df)

        assert components == {0: [0, 1]}


# No predicates  #


class TestNoPredicates:
    def test_returns_none_when_no_predicate_deduper(self):
        """A step without predicate dedupers resolves to `None`: the caller
        uses the signature branch, unchanged."""
        from liken.dedupers.exact import Exact

        step = [unit("col", Exact())]
        df = make_df({"col": ["a", "b", "a"]})

        assert resolve_visible_pool(step, df) is None


# Single predicate  #


class TestSinglePredicate:
    def test_full_range_when_every_row_matches(self):
        df = make_df({"col": ["x", "x", "x"]})
        step = [unit("col", IsIn(["x"]))]

        pool = resolve_visible_pool(step, df)

        assert pool == {0, 1, 2}

    def test_single_matcher_contributes_nothing(self):
        """A predicate matching exactly one row emits no pairs: the row is
        not a pair endpoint and stays invisible. Old behaviour agreed —
        single-member components never entered the union pool — and the
        downstream deduper cannot group an invisible row."""
        df = make_df({"col": ["x", "y", "y"]})
        step = [unit("col", IsIn(["x"]))]

        pool = resolve_visible_pool(step, df)

        assert pool == set()

    def test_match_set_is_indices_of_matching_rows(self):
        df = make_df({"col": ["x", "y", "x", "z"]})
        step = [unit("col", IsIn(["x"]))]

        pool = resolve_visible_pool(step, df)

        assert pool == {0, 2}

    def test_empty_match_set_yields_empty_pool(self):
        """A predicate matching nothing yields an empty pool: an empty
        match is a real empty, not a full-frame fallback. Old behaviour:
        the pool fell back to the full frame, so an empty-match predicate
        filtered nothing."""
        df = make_df({"col": ["a", "b"]})
        step = [unit("col", StrStartsWith("zzz"))]

        pool = resolve_visible_pool(step, df)

        assert pool == set()


# Multi-predicate intersection  #


class TestMultiPredicateIntersection:
    def test_intersection_excludes_rows_failing_either(self):
        """The pool is the running intersection of match sets. Old
        behaviour: the pool grew by union of multi-member components, so a
        row failing one predicate but entering via another stayed
        visible."""
        df = make_df({"c1": ["x", "x", "x", "y"], "c2": ["A", "B", "A", "A"]})
        step = [unit("c1", IsIn(["x"])), unit("c2", IsIn(["A"]))]

        pool = resolve_visible_pool(step, df)

        assert pool == {0, 2}

    def test_empty_intersection_is_empty(self):
        """Disjoint match sets intersect to the empty pool: boolean AND,
        no fallback. Row 0 fails `c2`, row 1 fails `c1`."""
        df = make_df({"c1": ["x", "y"], "c2": ["B", "A"]})
        step = [unit("c1", IsIn(["x"])), unit("c2", IsIn(["A"]))]

        pool = resolve_visible_pool(step, df)

        assert pool == set()

    def test_negated_predicate_participates_in_intersection(self):
        """Contributions are pool-restricted: the second predicate pairs
        only rows the first kept visible. c1 keeps {0,1}; within that,
        `~isna` sees one present row and emits no pairs. Old behaviour
        also evaluated predicates against the shrinking pool."""
        df = make_df({"c1": ["x", "x", "y", "y"], "c2": ["A", None, "A", None]})
        step = [unit("c1", IsIn(["x"])), unit("c2", ~IsNA())]

        pool = resolve_visible_pool(step, df)

        assert pool == set()


# Preprocessors  #


class TestPreprocessors:
    def test_preprocessors_apply_before_matching(self):
        """The preprocessed column drives pair generation. `ab` matches
        alone: one matching row emits no pairs, so the pool is empty."""
        from liken.preprocessors import lower

        df = make_df({"col": ["AB", "cd"]})
        step = [unit("col", IsIn(["ab"]), [lower()])]

        pool = resolve_visible_pool(step, df)

        assert pool == set()


# Pair-endpoint contributions  #


class TestPairEndpointContribution:
    """A predicate's contribution is the rows appearing in at least one
    pair it emits — its grouping membership, not its per-row match.

    Amended 2026-10-31: per-row matching cannot express relational
    predicates. `sorted_tokens`' per-row match is "has at least one
    token", but it groups by sorted-token bucket; `date_range`'s per-row
    match is "the value parses", but it groups by window cells and
    star-groups nulls. The pair-endpoint set is the grouping membership
    for both, and equals the per-row match set for star-shaped
    predicates (`isin`, `str_*`, `num_range`, `isna`).
    """

    def test_sorted_tokens_contributes_bucket_members_only(self):
        """A token-bearing row sharing no bucket with any other row is not
        a pair endpoint: invisible to downstream dedupers."""
        from liken.core.predicate_pushdown import resolve_visible_pool
        from liken.dedupers.sorted_tokens import SortedTokens

        df = make_df({"col": ["a b", "b a", "c d", "solo tokens", None]})
        step = [unit("col", SortedTokens())]

        pool = resolve_visible_pool(step, df)

        # rows 0,1 share the bucket ("a","b"); row 2 is a singleton
        # bucket; row 3 has tokens but no partner; row 4 is null
        assert pool == {0, 1}

    def test_date_range_contributes_window_cells_and_null_groups(self):
        """Nulls group together under `date_range`, though they never
        match per-row: they appear as pair endpoints."""
        from liken.core.predicate_pushdown import resolve_visible_pool
        from liken.dedupers.date_range import DateRange

        df = make_df(
            {
                "col": [
                    "2024-01-05",
                    "2024-01-07",
                    "2024-01-08",
                    None,
                    None,
                    "2024-06-01",
                ]
            }
        )
        step = [unit("col", DateRange(format="%Y-%m-%d", window=1))]

        pool = resolve_visible_pool(step, df)

        # Jan 7-Jan 8 within one day: rows 1,2. Nulls star-group: 3,4.
        # Jan 5 is 2 days from Jan 7 (window=1): not an endpoint.
        # Jun 1 alone: not an endpoint.
        assert pool == {1, 2, 3, 4}
