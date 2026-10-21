import math
import random
from datetime import date
from decimal import Decimal
from math import nextafter
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest
from rapidfuzz import fuzz

import liken as lk
from liken.constants import CANONICAL_ID
from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.dedupers.edit_distance import EditDistance
from liken.dedupers.exact import Exact
from liken.dedupers.fuzzy import Fuzzy
from liken.dedupers.isin import isin
from liken.dedupers.jaccard import Jaccard
from liken.dedupers.str_contains import StrContains
from liken.dedupers.str_startswith import StrStartsWith
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


############
# Fixtures #
###########


@pytest.fixture
def mock_df():
    """
    Minimal LocalDF. Only methods used by dedupers are defined.
    """
    df = Mock()
    df._get_col.return_value = pa.array([1, 2, 3])
    df._get_cols.return_value = pa.array([[1], [2], [3]])
    df.put_col.return_value = df
    df.get_array.return_value = pa.array([1, 2, 3])
    return df


##############################
# BaseDeduper core behavior #
##############################


def test_set_frame_sets_wrapped_df(mock_df):
    deduper = BaseDeduper()
    returned = deduper.set_frame(mock_df)
    assert returned is deduper
    assert deduper.wdf is mock_df


def test_gen_similarity_pairs_not_implemented():
    deduper = BaseDeduper()
    with pytest.raises(NotImplementedError):
        list(deduper._gen_similarity_pairs(pa.array([])))


################
# canonicalize #
################


def test_canonicalize_puts_canonical_id(mock_df):
    deduper = BaseDeduper()
    deduper.set_frame(mock_df)

    deduper.wdf.get_array = Mock(
        side_effect=[
            pa.array([10, 20, 30]),
            pa.array(["a", "a", "b"]),
        ]
    )

    deduper.wdf.get_canonical = Mock(side_effect=[pa.array([10, 20, 30])])

    components = {
        0: [0, 1],
        2: [2],
    }

    result = deduper.canonicalizer(
        components=components, drop_duplicates=False, keep="first", canonical_id=CANONICAL_ID
    )

    mock_df.put_col.assert_called_once()
    assert result is mock_df


####################
# ColumnArrayMixin #
####################


def test_column_array_mixin_str_column(mock_df):
    deduper = Exact().set_frame(mock_df)
    arr = deduper.wdf.get_array("a")
    mock_df.get_array.assert_called_once_with("a")
    assert isinstance(arr, pa.Array)


def test_column_array_mixin_tuple_column(mock_df):
    deduper = Exact().set_frame(mock_df)
    arr = deduper.wdf.get_array(("a", "b"))
    mock_df.get_array.assert_called_once_with(("a", "b"))
    assert isinstance(arr, pa.Array)


#####################
# Validation mixins #
#####################


def test_single_column_validation_accepts_str():
    StrStartsWith("x").validate("col")


def test_single_column_validation_rejects_tuple():
    with pytest.raises(InvalidArgumentTypeError):
        StrStartsWith("x").validate(("a", "b"))


def test_compound_column_validation_accepts_tuple():
    Jaccard().validate(("a", "b"))


def test_compound_column_validation_rejects_str():
    with pytest.raises(InvalidArgumentTypeError):
        Jaccard().validate("a")


####################
# str/repr output #
###################


STR_PARAMS = [
    ("edit_distance", lk.edit_distance(), ["edit_distance(", "max_distance=2"]),
    ("exact", lk.exact(), ["exact()"]),
    ("fuzzy", lk.fuzzy(), ["fuzzy(", "threshold=0.95"]),
    ("tfidf", lk.tfidf(ngram=1, topn=2), ["tfidf(", "threshold=0.95", "ngram=1", "topn=2"]),
    ("lsh", lk.lsh(ngram=1, num_perm=128), ["lsh(", "threshold=0.95", "ngram=1", "num_perm=128"]),
    (
        "str_startswith",
        lk.str_startswith(pattern="calle", case=False),
        ["str_startswith(", "pattern='calle'", "case=False"],
    ),
    (
        "str_endswith",
        lk.str_endswith(pattern="kingdom", case=False),
        ["str_endswith(", "pattern='kingdom'", "case=False"],
    ),
    (
        "str_contains",
        lk.str_contains(pattern="05", case=False, regex=True),
        ["str_contains(", "pattern='05'", "case=False", "regex=True"],
    ),
    ("str_len", lk.str_len(), ["str_len(", "min_len=0", "max_len=None"]),
    ("cosine", lk.cosine(), ["cosine(", "threshold=0.95"]),
    ("jaccard", lk.jaccard(threshold=0.5), ["jaccard(", "threshold=0.5"]),
    ("isin", lk.isin(["london"]), ["isin(", "values=('london',)"]),
    ("isna", lk.isna(), ["isna()"]),
    ("~isna", ~lk.isna(), ["~isna()"]),
    (
        "~str_startswith",
        ~lk.str_startswith(pattern="calle", case=False),
        ["~str_startswith(", "pattern='calle'", "case=False"],
    ),
]


@pytest.mark.parametrize("name, deduper, expected_parts", STR_PARAMS, ids=[p[0] for p in STR_PARAMS])
def test_deduper_str_names_deduper_and_args(name, deduper, expected_parts):
    representation = str(deduper)

    assert representation  # non-empty
    assert all(part in representation for part in expected_parts)


def test_base_deduper_str_falls_back_to_repr():
    deduper = BaseDeduper()
    assert str(deduper) == repr(deduper)
    assert str(deduper) == "BaseDeduper()"


def test_custom_deduper_str_names_function():
    @lk.custom.register
    def str_test_deduper(array):
        yield 0, 1

    representation = str(str_test_deduper())
    assert representation.startswith("_Custom(")
    assert "str_test_deduper" in representation


def test_custom_deduper_rejects_positional_args():
    @lk.custom.register
    def positional_test_deduper(array):
        yield 0, 1

    with pytest.raises(TypeError, match="positional_test_deduper must be called with keyword arguments only"):
        positional_test_deduper(1)


def test_custom_deduper_generates_pairs(mock_df):
    @lk.custom.register
    def pair_test_deduper(array):
        yield 0, 1

    deduper = pair_test_deduper()
    deduper.set_frame(mock_df)

    uf, n = deduper.build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[1]
    assert uf[0] != uf[2]


######################################
# Predicate deduper Python fallback #
#####################################


def test_negated_predicate_deduper_fallback_matching(mock_df):
    """~isin has no vectorised form, so it exercises the negated deduper's
    Python fallback path."""
    deduper = ~isin([2])  # negate-match: everything except the value 2
    deduper.set_frame(mock_df)  # wdf.get_array -> pa.array([1, 2, 3])

    uf, n = deduper.build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[2]  # 1 and 3 negate-match each other
    assert uf[0] != uf[1]  # 2 does not match them


############################################
# Negated predicates never match missing  #
###########################################


@pytest.mark.parametrize("values", [["a"], ["a", None]], ids=["null-unlisted", "null-listed"])
def test_negated_isin_never_matches_null(mock_df, values):
    """A missing value never satisfies a negated predicate.

    The null stays out of every group, whether or not None is in `values`:
    the negation must not consult the inner membership test for a missing
    value.
    """
    mock_df.get_array = Mock(return_value=pa.array([None, "a", None, "b"]))

    uf, n = (~isin(values)).set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[0] != uf[1] and uf[0] != uf[3]  # null vs each value
    assert uf[0] != uf[2]  # nulls do not group under a negated predicate
    assert uf[2] != uf[1] and uf[2] != uf[3]


def test_negated_isin_never_matches_nan(mock_df):
    """NaN is missing and never satisfies a negated predicate."""
    mock_df.get_array = Mock(return_value=pa.array([float("nan"), 1.0, 2.0]))

    uf, n = (~isin([1.0])).set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] != uf[1] and uf[0] != uf[2]


def test_negated_str_startswith_vectorized_skips_nulls():
    """The vectorised negated path leaves missing values unmatched."""
    array = pa.array(["alpha", None, "beta", "gamma"])

    pairs = list((~StrStartsWith(pattern="al"))._gen_similarity_pairs(array))

    assert pairs == [(2, 3)]  # "beta" and "gamma" only; the null is skipped


@pytest.mark.parametrize(
    "values, null_grouped",
    [(["a"], False), (["a", None], True)],
    ids=["null-unlisted", "null-listed"],
)
def test_isin_null_matches_iff_none_is_listed(mock_df, values, null_grouped):
    """Positive isin keeps Python membership: a null matches iff None is in values."""
    mock_df.get_array = Mock(return_value=pa.array([None, "a", None, "b"]))

    uf, n = isin(values).set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    if null_grouped:
        assert uf[0] == uf[1] == uf[2]  # None listed: the nulls match
        assert uf[0] != uf[3]
    else:
        assert uf[0] != uf[1] and uf[0] != uf[2] and uf[1] != uf[2]  # only "a" matches


#################################
# exact bucket keys             #
#################################


def test_exact_nulls_pair_with_nulls_only(mock_df):
    """A null groups only with another null; literal "na" is an ordinary value."""
    mock_df.get_array = Mock(return_value=pa.array(["na", None, "na", None]))

    uf, n = lk.exact().set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[0] == uf[2]  # the two "na" literals group as values
    assert uf[1] == uf[3]  # the two nulls group
    assert uf[0] != uf[1]  # a literal never groups with a null


@pytest.mark.parametrize("values", [[7, None, 7, None], [1.5, None, 1.5, None]], ids=["int", "float"])
def test_exact_non_string_columns_group_equal_values(mock_df, values):
    """Non-string columns group equal values; nulls group with nulls."""
    mock_df.get_array = Mock(return_value=pa.array(values))

    uf, n = lk.exact().set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[0] == uf[2]
    assert uf[1] == uf[3]
    assert uf[0] != uf[1]


def test_exact_nan_groups_with_null(mock_df):
    """NaN is missing on the same footing as None: one missing group."""
    mock_df.get_array = Mock(return_value=pa.array([1.0, None, float("nan"), None]))

    uf, n = lk.exact().set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[1] == uf[2] == uf[3]  # the two nulls and the NaN form one group
    assert uf[0] != uf[1]


def test_exact_nested_typed_column_groups_by_content(mock_df):
    """Nested-typed columns keep grouping by content.

    Bucket keys stay pa.Scalar, which is hashable for list/struct types;
    normalising through as_py() would raise TypeError on the unhashable
    Python list or dict a nested scalar converts to.
    """
    mock_df.get_array = Mock(return_value=pa.array([[1], [1], [2]], type=pa.list_(pa.int64())))

    uf, n = lk.exact().set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[1]  # equal contents group
    assert uf[0] != uf[2]


def test_exact_compound_missing_members_collapse(mock_df):
    """Missing members collapse per member: (None, "x") and (NaN, "x") are one key."""
    mock_df.get_array = Mock(
        return_value=pa.table({"a": pa.array([None, float("nan"), None]), "b": pa.array(["x", "x", "y"])})
    )

    uf, n = lk.exact().set_frame(mock_df).build_union_find(("a", "b"), [])

    assert n == 3
    assert uf[0] == uf[1]  # (None, "x") groups with (NaN, "x")
    assert uf[0] != uf[2]  # (None, "x") does not group with (None, "y")


##################################
# isna NaN and None semantics #
#################################


def test_isna_groups_none_and_nan_together(mock_df):
    mock_df.get_array = Mock(return_value=pa.array([None, float("nan"), 1.0]))

    uf, n = lk.isna().set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[1]  # None and NaN group together
    assert uf[0] != uf[2]


def test_notna_groups_only_non_null_values(mock_df):
    mock_df.get_array = Mock(return_value=pa.array([None, float("nan"), 1.0, 2.0]))

    uf, n = (~lk.isna()).set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[2] == uf[3]  # only the non-null values group
    assert uf[0] != uf[2] and uf[1] != uf[2]


##############################
# fuzzy null partitioning   #
#############################


def test_fuzzy_nulls_pair_with_nulls_only(mock_df):
    """Missing values pair with each other, never with a value.

    fuzz.ratio("nathan", "nadia") is ~54.5, so the two values also pair at a
    0.5 threshold; the nulls form their own group either way.
    """
    mock_df.get_array = Mock(return_value=pa.array(["nathan", None, "nadia", None]))

    uf, n = lk.fuzzy(threshold=0.5).set_frame(mock_df).build_union_find("address", [])

    assert n == 4
    assert uf[1] == uf[3]  # the two nulls group
    assert uf[0] == uf[2]  # the values group at this threshold
    assert uf[0] != uf[1]  # a null never groups with a value


@pytest.mark.parametrize(
    "threshold, expected_pairs",
    [(math.nextafter(0.0, 1.0), [(0, 2), (1, 3)]), (0.55, [(1, 3)])],
    ids=["near-zero-threshold", "just-above-the-value-pair"],
)
def test_fuzzy_null_never_pairs_with_value(threshold, expected_pairs):
    """At any threshold a null pairs only with another null."""
    array = pa.array(["nathan", None, "nadia", None])

    pairs = sorted(lk.fuzzy(threshold=threshold)._gen_similarity_pairs(array))

    assert pairs == expected_pairs


def test_fuzzy_all_null_input_yields_star_pairs():
    """All-null input yields star-shaped null pairs without touching the scorer."""
    array = pa.array([None, None, None])

    assert list(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


def test_fuzzy_nan_and_null_pair():
    """NaN is missing and pairs with a null, never with a value."""
    array = pa.array([float("nan"), 1.0, None])

    assert list(lk.fuzzy(threshold=0.5)._gen_similarity_pairs(array)) == [(0, 2)]


def test_fuzzy_cdist_receives_no_missing_values(monkeypatch):
    """process.cdist scores non-missing values only."""
    from rapidfuzz import process

    from liken.core.missing import is_missing

    real_cdist = process.cdist

    def spy_cdist(queries, choices, **kwargs):
        assert all(not is_missing(value) for value in (*queries, *choices))
        return real_cdist(queries, choices, **kwargs)

    monkeypatch.setattr(process, "cdist", spy_cdist)

    array = pa.array(["nathan", None, "nadia", None])

    assert list(lk.fuzzy(threshold=0.95)._gen_similarity_pairs(array)) == [(1, 3)]


##########################
# Vectorized mask guard #
##########################


class _FallbackOnlyStrContains(StrContains):
    def _vectorized_matches(self, array: pa.Array) -> pa.Array | None:
        return None


def test_vectorized_guard_dispatches_on_presence_not_truthiness(mock_df):
    """The guard must read the mask's identity, not its truthiness.

    A subclass may return an empty mask (falsy under pyarrow's len-based
    ``__bool__``); the vectorized path must still be taken.
    """

    class EmptyMaskDeduper(PredicateDeduper):
        def _vectorized_matches(self, array: pa.Array) -> pa.Array | None:
            return pa.array([], type=pa.bool_())

        def _matches(self, value):
            raise AssertionError("Python fallback must not run when a mask is present")

    deduper = EmptyMaskDeduper().set_frame(mock_df)

    assert list(deduper._gen_similarity_pairs(pa.array(["a", "b"], type=pa.string()))) == []


@pytest.mark.parametrize("values", [["apple"], []], ids=["one-row", "zero-rows"])
def test_vectorized_matches_same_pairs_as_python_fallback(values):
    array = pa.array(values, type=pa.string())

    vectorized = StrContains(pattern="app")
    fallback = _FallbackOnlyStrContains(pattern="app")

    assert list(vectorized._gen_similarity_pairs(array)) == list(fallback._gen_similarity_pairs(array))


def test_fuzzy_unknown_scorer_raises_at_construction():
    """an unknown scorer is rejected at construction"""
    with pytest.raises(InvalidArgumentValueError, match="`scorer` must be one of"):
        Fuzzy(threshold=0.95, scorer="no_such_scorer")


def test_fuzzy_rejects_non_str_scorer():
    with pytest.raises(InvalidArgumentTypeError, match="`scorer` must be of type 'str', got 'int'"):
        lk.fuzzy(scorer=1)


def test_fuzzy_jaro_winkler_scorer_maps_onto_0_100_scale():
    """The registry wrapper scales normalized_similarity to the 0-100 scale.

    Raw `JaroWinkler.normalized_similarity` returns 0-1; the deduper
    compares scores against `100 * threshold`, so the registry value must
    return 100.0 for identical values and 0.0 for disjoint ones.
    """
    scorer = Fuzzy(threshold=0.95, scorer="jaro_winkler").get_scorer()

    assert scorer("abc", "abc") == 100.0
    assert scorer("abc", "xyz") == 0.0


def test_fuzzy_jaro_winkler_matches_real_pair_at_scaled_threshold():
    """A 0.905 pair matches at threshold 0.9 and not at 0.91.

    The fixture pair scores 0.905041 with JaroWinkler. A raw 0-1 score
    compared unconverted against `100 * threshold` (90, 91) would match
    neither, so the threshold=0.9 case fails if the scale conversion is
    dropped; the 0.91 case then separates a converted 90.5 from a
    threshold of 91.
    """
    s1 = "Calle Ancho, 12, 05688, Rioja, Navarra, Espana"
    s2 = "Calle Sueco, 56, 05688, Rioja, Navarra"

    pairs = list(lk.fuzzy(threshold=0.9, scorer="jaro_winkler")._gen_similarity_pairs(pa.array([s1, s2])))

    assert pairs == [(0, 1)]

    pairs = list(lk.fuzzy(threshold=0.91, scorer="jaro_winkler")._gen_similarity_pairs(pa.array([s1, s2])))

    assert pairs == []


def test_fuzzy_jaro_winkler_identical_pair_matches_high_threshold_mismatched_does_not():
    """Threshold is validated in the open interval (0, 1), so both probes stay in it."""
    deduper = lk.fuzzy(threshold=0.99, scorer="jaro_winkler")

    assert list(deduper._gen_similarity_pairs(pa.array(["abc", "abc"]))) == [(0, 1)]

    deduper = lk.fuzzy(threshold=0.01, scorer="jaro_winkler")

    assert list(deduper._gen_similarity_pairs(pa.array(["abc", "xyz"]))) == []


def test_fuzzy_default_scorer_is_unchanged():
    assert lk.fuzzy().get_scorer() is fuzz.ratio

    # the existing six scorers still construct
    for name in [
        "simple_ratio",
        "partial_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "weighted_ratio",
        "quick_ratio",
    ]:
        assert lk.fuzzy(scorer=name).get_scorer() is not None


###########################
# fuzzy topn acceptance  #
##########################

# WI050 (knn) mirrors these acceptance-semantics tests: keep the
# `test_fuzzy_topn_*` names and the pinned semantics identical.


def _stub_score_table(table):
    """A scorer stub driven by a value-keyed score table.

    Accepts `score_cutoff` as a kwarg: `process.extract` passes it. The
    stub returns the table score for the pair, or 0.0 for unlisted pairs.
    """

    def scorer(s1, s2, score_cutoff=None):
        return table.get((s1, s2), table.get((s2, s1), 0.0))

    return scorer


def test_fuzzy_topn_none_is_all_pairs():
    """`topn=None` (the default) is all-pairs fuzzy.

    With topn=1 the pair set on this fixture would differ: "abcf" is only
    the third-best partner of "abcd" and would be dropped.
    """
    array = pa.array(["abcd", "abce", "abcf", "zzzz"])

    default_pairs = sorted(lk.fuzzy(threshold=0.7)._gen_similarity_pairs(array))

    assert default_pairs == [(0, 1), (0, 2), (1, 2)]

    explicit_none_pairs = sorted(lk.fuzzy(threshold=0.7, topn=None)._gen_similarity_pairs(array))

    assert explicit_none_pairs == default_pairs


def test_fuzzy_accepts_topn_none():
    """`topn=None` constructs: None means the all-pairs default.

    Deliberately unlike tfidf, which rejects `topn=None` (its default 2 is
    an active value, not "absent").
    """
    lk.fuzzy(topn=None)


def test_fuzzy_topn_either_direction_emits_pair(monkeypatch):
    """A pair is emitted when either side holds the other in its k best.

    Score table: s(A,B)=80, s(A,C)=70, s(B,C)=60; topn=1, threshold 0.75.
    A's best is B (80), B's best is A (80): pair (0, 1). C's best is A
    (70, below threshold): no pair via C.

    At threshold 0.55, C's best (A at 70) clears the threshold while A's
    best is still B: pair (0, 2) survives through C's direction alone —
    the asymmetric case a one-direction-per-pair scan would miss.
    """
    scores = {("A", "B"): 80.0, ("A", "C"): 70.0, ("B", "C"): 60.0}

    deduper = lk.fuzzy(threshold=0.75, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["A", "B", "C"])))

    assert pairs == [(0, 1)]

    deduper = lk.fuzzy(threshold=0.55, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["A", "B", "C"])))

    assert pairs == [(0, 1), (0, 2)]


def test_fuzzy_topn_k_membership_is_per_row(monkeypatch):
    """Each row keeps its own k best partners.

    Scores: s(A,B)=80, s(A,C)=75, s(A,D)=70; s(B,C)=s(B,D)=s(C,D)=10.
    topn=1, threshold 0.6: A's best is B only; C's best is A (75) so
    (0, 2) survives via C's direction; D's best is A (70) so (0, 3)
    survives too. topn is per row: one row's k does not bound another
    row's partner count.
    """
    scores = {
        ("A", "B"): 80.0,
        ("A", "C"): 75.0,
        ("A", "D"): 70.0,
        ("B", "C"): 10.0,
        ("B", "D"): 10.0,
        ("C", "D"): 10.0,
    }

    deduper = lk.fuzzy(threshold=0.6, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["A", "B", "C", "D"])))

    assert pairs == [(0, 1), (0, 2), (0, 3)]


@pytest.mark.parametrize(
    "drift, expected",
    [(0.0, [(0, 1)]), (-5e-10, [(0, 1)]), (-2e-9, [])],
    ids=["exact-at-threshold", "within-tolerance", "beyond-tolerance"],
)
def test_fuzzy_topn_boundary_drift(monkeypatch, drift, expected):
    """The top-k path shares the all-pairs boundary contract.

    The stub returns `100 * threshold + drift` for every pair. At exact
    boundary and within the 1e-9 tolerance the pair is emitted; beyond the
    tolerance nothing matches — and nothing below the band can appear in a
    top-k list, because extract's cutoff already filters it.
    """
    threshold = 0.75
    score = float(Decimal(str(threshold)) * 100) + drift

    deduper = lk.fuzzy(threshold=threshold, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: lambda s1, s2, score_cutoff=None: score)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["x", "y"])))

    assert pairs == expected


def test_fuzzy_topn_matches_a_real_score_exactly_at_the_threshold():
    """The native-scorer cutoff path matches exactly at the threshold.

    `fuzz.ratio("abcd", "abce")` is exactly 75.0: threshold 0.75 matches
    and 0.76 does not, through `process.extract`'s own cutoff comparison.
    """
    assert fuzz.ratio("abcd", "abce") == 75.0

    pairs = list(lk.fuzzy(threshold=0.75, topn=1)._gen_similarity_pairs(pa.array(["abcd", "abce"])))

    assert pairs == [(0, 1)]

    pairs = list(lk.fuzzy(threshold=0.76, topn=1)._gen_similarity_pairs(pa.array(["abcd", "abce"])))

    assert pairs == []


def test_fuzzy_topn_ties_broken_by_index_order():
    """Ties at rank k resolve by choice index, not by permutation invariance.

    Five identical values, topn=2, threshold 0.99: every score is 100, so
    each row's k best are the two lowest-index other rows. The undirected
    pair set is {(0,1), (0,2), (1,2), (0,3), (1,3), (0,4), (1,4)} — 7
    pairs. Reordering the input can change the result: the same input
    order gives the same output, nothing more (WI017 concern).
    """
    array = pa.array(["A"] * 5)

    pairs = sorted(lk.fuzzy(threshold=0.99, topn=2)._gen_similarity_pairs(array))

    assert pairs == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]


def test_fuzzy_topn_exceeding_partners_matches_all_pairs():
    """k >= n - 1 degenerates to all-pairs."""
    array = pa.array(["abcd", "abce", "abcf"])

    all_pairs = sorted(lk.fuzzy(threshold=0.7)._gen_similarity_pairs(array))
    topn_pairs = sorted(lk.fuzzy(threshold=0.7, topn=10)._gen_similarity_pairs(array))

    assert all_pairs == [(0, 1), (0, 2), (1, 2)]
    assert topn_pairs == all_pairs


def test_fuzzy_topn_jaro_winkler_matches_at_scaled_threshold():
    """The jaro_winkler wrapper works under the top-k path too.

    `process.extract` passes a real `score_cutoff` to the wrapper, which
    ignores it and returns the full 0-100 score; extract then compares
    that score against the cutoff itself. The 0.905 pair matches at
    threshold 0.9 and not at 0.91, same as the all-pairs path.
    """
    s1 = "Calle Ancho, 12, 05688, Rioja, Navarra, Espana"
    s2 = "Calle Sueco, 56, 05688, Rioja, Navarra"

    pairs = list(lk.fuzzy(threshold=0.9, scorer="jaro_winkler", topn=1)._gen_similarity_pairs(pa.array([s1, s2])))

    assert pairs == [(0, 1)]

    pairs = list(lk.fuzzy(threshold=0.91, scorer="jaro_winkler", topn=1)._gen_similarity_pairs(pa.array([s1, s2])))

    assert pairs == []


def test_fuzzy_topn_nulls_star_pair_outside_topk():
    """Missing values keep the star-shaped handling, outside the top-k path.

    fuzz.ratio("a", "ab") is ~66.7: no value pair at threshold 0.95. The
    null star pair (1, 3) is unaffected by topn.
    """
    array = pa.array(["a", None, "ab", None])

    pairs = sorted(lk.fuzzy(threshold=0.95, topn=1)._gen_similarity_pairs(array))

    assert pairs == [(1, 3)]


def test_fuzzy_topn_all_null_input_yields_star_pairs():
    array = pa.array([None, None, None])

    assert list(lk.fuzzy(threshold=0.5, topn=1)._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


def test_fuzzy_topn_extract_receives_no_missing_values(monkeypatch):
    """process.extract scores non-missing values only."""
    from rapidfuzz import process

    from liken.core.missing import is_missing

    real_extract = process.extract

    def spy_extract(query, choices, **kwargs):
        assert all(not is_missing(value) for value in (query, *choices))
        return real_extract(query, choices, **kwargs)

    monkeypatch.setattr(process, "extract", spy_extract)

    array = pa.array(["nathan", None, "nadia", None])

    assert list(lk.fuzzy(threshold=0.95, topn=1)._gen_similarity_pairs(array)) == [(1, 3)]


def test_fuzzy_topn_transitivity_through_union_find(mock_df, monkeypatch):
    """Merged pairs chain transitively through union-find.

    Scores: s(A,B)=80, s(A,C)=60, s(B,C)=75; topn=1, threshold 0.7.
    Pairs (A,B) and (B,C) qualify; A-C never does. Union-find still merges
    all three through B — the docstring's transitivity note in action.
    """
    scores = {("A", "B"): 80.0, ("A", "C"): 60.0, ("B", "C"): 75.0}

    deduper = lk.fuzzy(threshold=0.7, topn=1)
    monkeypatch.setattr(deduper, "get_scorer", lambda: _stub_score_table(scores))

    mock_df.get_array = Mock(return_value=pa.array(["A", "B", "C"]))

    uf, n = deduper.set_frame(mock_df).build_union_find("address", [])

    assert n == 3
    assert uf[0] == uf[1] == uf[2]


def test_fuzzy_topn_calls_extract_once_per_present_row(monkeypatch):
    """The mechanism is candidate-limited per row, not a trimmed full matrix.

    `process.extract` runs exactly once per present row, each call with
    `limit=k+1` and the full present-value list as choices.
    """
    from rapidfuzz import process

    real_extract = process.extract
    calls = []

    def spy_extract(query, choices, **kwargs):
        calls.append({"limit": kwargs.get("limit"), "choices_len": len(choices)})
        return real_extract(query, choices, **kwargs)

    monkeypatch.setattr(process, "extract", spy_extract)

    array = pa.array(["abcd", "abce", "abcf", None])

    pairs = list(lk.fuzzy(threshold=0.7, topn=2)._gen_similarity_pairs(array))

    assert pairs == [(0, 1), (0, 2), (1, 2)]
    assert len(calls) == 3  # one per present row
    assert all(call["limit"] == 3 for call in calls)  # k + 1
    assert all(call["choices_len"] == 3 for call in calls)  # full present list


@pytest.mark.parametrize("topn", [0, -1])
def test_fuzzy_rejects_non_positive_topn(topn):
    """`topn` below 1 is refused at construction: an unvalidated topn reaches
    the native library and can abort the process."""
    with pytest.raises(InvalidArgumentValueError, match="`topn` must be an integer in the range \\[1, inf\\), got"):
        lk.fuzzy(topn=topn)


@pytest.mark.parametrize("topn", [2.0, "2", True, []])
def test_fuzzy_rejects_non_integer_topn(topn):
    """`topn` must be a positive integer or None.

    tfidf rejects None (its default 2 is an active value); fuzzy
    deliberately accepts it — None is the all-pairs default here.
    """
    with pytest.raises(InvalidArgumentTypeError, match="`topn` must be an integer, got"):
        lk.fuzzy(topn=topn)


def test_fuzzy_topn_str_repr_omitted_when_none():
    """The default repr stays `fuzzy(threshold=0.95)`; a set topn renders."""
    assert "topn" not in str(lk.fuzzy())
    assert "topn=5" in str(lk.fuzzy(topn=5))


################
# knn          #
################

# Mirrors the `test_fuzzy_topn_*` acceptance block (WI049): one
# mechanism, one policy — threshold required ∩ either-direction top-k.
# The tie test is the deliberate divergence: cKDTree's tie order is its
# own, not index order (WI049 D3 analog differs).


def test_knn_default_metric_is_euclidean():
    assert lk.knn(k=5)._metric == "euclidean"


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_knn_accepts_each_metric(metric):
    assert lk.knn(k=5, metric=metric)._metric == metric


def test_knn_unknown_metric_raises_value_error():
    with pytest.raises(InvalidArgumentValueError, match="`metric` must be one of"):
        lk.knn(k=5, metric="manhattan")


def test_knn_non_str_metric_raises_type_error():
    with pytest.raises(InvalidArgumentTypeError, match="`metric` must be of type 'str', got 'int'"):
        lk.knn(k=5, metric=1)


@pytest.mark.parametrize("k", [0, -1])
def test_knn_rejects_non_positive_k(k):
    """`k` below 1 is refused at construction: an unvalidated k reaches
    cKDTree and can abort the process."""
    with pytest.raises(InvalidArgumentValueError, match="`k` must be an integer in the range \\[1, inf\\), got"):
        lk.knn(k=k)


@pytest.mark.parametrize("k", [2.0, "2", True, None, []])
def test_knn_rejects_non_integer_k(k):
    """`k` must be a positive integer; there is no default.

    Unlike fuzzy's `topn`, `None` is rejected: `k` is required and has no
    absent state.
    """
    with pytest.raises(InvalidArgumentTypeError, match="`k` must be an integer, got"):
        lk.knn(k=k)


def test_knn_str_renders_threshold_k_and_metric():
    """`k` and `metric` always render: `k` has no absent state and
    `metric` has no default-free None."""
    assert "knn(threshold=0.95, k=5, metric='euclidean')" in str(lk.knn(k=5))
    assert "metric='cosine'" in str(lk.knn(k=5, metric="cosine"))


def test_knn_requires_compound_tuple_columns():
    with pytest.raises(InvalidArgumentTypeError):
        lk.knn(k=5).validate("name")

    with pytest.raises(InvalidArgumentTypeError):
        lk.knn(k=5).validate(("name", 3))


def test_knn_table_with_two_columns_pairs_close_rows():
    """Rows (0, 0) and (1, 0) sit at distance 1; rows 0 and 2 sit at
    distance 3. The euclidean similarity is 1/(1+d): 0.5 for the close
    pair, 0.25 for the far pair. At threshold 0.45 only the close pair
    matches."""
    table = pa.table({"x": [1.0, 2.0, 1.0], "y": [1.0, 1.0, 4.0]})

    pairs = list(lk.knn(threshold=0.45, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_knn_either_direction_emits_pair():
    """A pair is emitted when either side holds the other in its k best.

    Points on a line: A=0, B=1, C=10, k=1. A's best is B (d=1, s=0.5),
    B's best is A (d=1), C's best is B (d=9, s=0.1). At threshold 0.3,
    only (A,B) clears. At threshold 0.09, C's best partner B clears it
    (0.1) while B's best is still A: pair (B,C) survives through C's
    direction alone — the asymmetric case a one-direction-per-pair scan
    would miss. A does not hold C at any threshold here: C is not A's
    nearest.
    """
    table = pa.table({"x": [1.0, 2.0, 11.0]})

    pairs = sorted(lk.knn(threshold=0.3, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]

    pairs = sorted(lk.knn(threshold=0.09, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2)]


def test_knn_k_membership_is_per_row():
    """Each row keeps its own k best partners.

    Points: A=1, B=2, C=2.5, D=3.5, k=1. A's best is B (d=1, s=0.5), B's
    best is C (d=0.5, s ~ 0.667), C's best is B, D's best is C (d=1,
    s=0.5). At threshold 0.45, A-B, B-C and C-D all clear:
    chain emission. C-D survives through D's direction only — C's best
    is B, so a one-direction-per-pair scan from C would miss it. With
    k=2 the set is unchanged: A's second partner C sits at s=0.4, below
    the threshold. k is per row, and no row's k bounds another row's
    partner count.
    """
    table = pa.table({"x": [1.0, 2.0, 2.5, 3.5]})

    pairs = sorted(lk.knn(threshold=0.45, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2), (2, 3)]

    pairs = sorted(lk.knn(threshold=0.45, k=2)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize(
    "threshold, expected",
    [(0.5, [(0, 1)]), (nextafter(0.5, 0.0), [(0, 1)]), (nextafter(0.5, 1.0), [])],
    ids=["exact-at-threshold", "just-below", "just-above"],
)
def test_knn_euclidean_boundary_in_the_similarity_domain(threshold, expected):
    """The euclidean comparison runs in the similarity domain: s = 1/(1+d).

    Two rows one unit apart give s = 0.5 exactly (the conversion is
    exact here): the threshold 0.5 matches, a threshold one ULP below
    also matches (0.5 >= it), a threshold one ULP above does not.
    Bare `>=` against the computed similarity, mirroring cosine: no
    tolerance band, the same inclusive-boundary contract as the compound
    numeric family.
    """
    table = pa.table({"x": [1.0, 2.0]})

    pairs = list(lk.knn(threshold=threshold, k=1)._gen_similarity_pairs(table))

    assert pairs == expected


def test_knn_cosine_boundary():
    """The cosine comparison runs on the pre-normalized conversion s = 1 - d^2/2.

    Rows [1, 0] and [0, 1] are orthogonal: after normalization d = sqrt(2),
    s = 1 - 2/2 = 0. The pair never matches on any positive threshold. Rows
    [1, 1] and [1, 0] have cosine similarity 1/sqrt(2): threshold equal to
    it matches. The documented numerical caveat bites at one ULP: the
    sqrt-then-square round trip through cKDTree leaves the converted
    similarity one ULP above the true cosine value, so the pair survives
    a one-ULP-higher threshold — one ULP looser than `lk.cosine`'s pinned
    boundary. Two ULPs above rejects, pinning that the drift is bounded
    at one ULP.
    """
    orthogonal = pa.table({"x": [1.0, 0.0], "y": [0.0, 1.0]})

    assert list(lk.knn(threshold=0.01, k=1, metric="cosine")._gen_similarity_pairs(orthogonal)) == []

    similar = pa.table({"x": [1.0, 1.0], "y": [1.0, 0.0]})
    exact = float(1.0 / np.sqrt(2.0))

    pairs = list(lk.knn(threshold=exact, k=1, metric="cosine")._gen_similarity_pairs(similar))

    assert pairs == [(0, 1)]

    one_up = float(math.nextafter(exact, math.inf))
    pairs = list(lk.knn(threshold=one_up, k=1, metric="cosine")._gen_similarity_pairs(similar))

    assert pairs == [(0, 1)]

    two_up = float(math.nextafter(one_up, math.inf))
    pairs = list(lk.knn(threshold=two_up, k=1, metric="cosine")._gen_similarity_pairs(similar))

    assert pairs == []


def test_knn_cosine_matches_all_pairs_cosine_at_large_k():
    """k >= n-1 degenerates to all-pairs; cosine knn matches lk.cosine exactly.

    The cross-check that the pre-normalized conversion reproduces cosine
    similarity on a non-boundary threshold (D2).
    """
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(12, 3))
    table = pa.table({f"col_{i}": matrix[:, i].tolist() for i in range(3)})

    threshold = 0.5

    knn_pairs = sorted(lk.knn(threshold=threshold, k=11, metric="cosine")._gen_similarity_pairs(table))
    cosine_pairs = sorted(lk.cosine(threshold=threshold)._gen_similarity_pairs(table))

    assert knn_pairs == cosine_pairs


def test_knn_ties_pinned_behaviour():
    """Ties at rank k follow cKDTree's own order, not index order.

    Five identical rows, k=2, threshold 0.99: every pair sits at d=0,
    s=1.0. cKDTree's tie order is deterministic per input but not
    index-ordered — every row's returned neighbours are rows 1 and 2
    (probed on scipy 1.16.3), so row 0 pairs with 1 and 2, row 1 with 2
    and 0, and rows 3 and 4 each reach only rows 1 and 2 within their
    k best. The pinned set is what scipy 1.16.3 returns; a scipy change
    that alters it fails this test visibly. Reordering the input can
    change the result: the same input order gives the same output,
    nothing more (WI017 concern). This is the deliberate divergence from
    the WI049 mirror: rapidfuzz breaks ties by index, cKDTree does not.
    """
    table = pa.table({"x": [1.0] * 5})

    pairs = sorted(lk.knn(threshold=0.99, k=2)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]


def test_knn_exceeding_partners_matches_all_pairs():
    """k >= n - 1 degenerates to all-pairs.

    Pairwise similarities 1/(1+d): d(0,1)=1 -> 0.5; d(0,2)=3 -> 0.25;
    d(1,2)=sqrt(10) ~ 3.162 -> 0.2403. All three clear 0.2.
    """
    table = pa.table({"x": [1.0, 2.0, 1.0], "y": [1.0, 1.0, 4.0]})

    all_pairs = sorted(lk.knn(threshold=0.2, k=10)._gen_similarity_pairs(table))

    assert all_pairs == [(0, 1), (0, 2), (1, 2)]


def test_knn_zero_norm_rows_match_nothing():
    """An all-zero row matches nothing, under both metrics.

    The spec pins "a row whose norm is 0 is normalised as a zero vector,
    so it matches nothing"; the clamp alone would put two zero rows at
    d=0, s=1.0 under both conversions, so zero-norm rows are kept out of
    the tree entirely: they neither emit nor appear in another row's k
    best. Here rows 0 and 1 are all-zero and excluded; row 2 never pairs
    with them even though euclidean d(row 2, row 0) = sqrt(2) would
    clear the threshold, and under cosine the clamped row would sit at
    similarity 0.5 from everything.
    """
    all_zero = pa.table({"x": [0.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0]})

    for metric in ("euclidean", "cosine"):
        pairs = list(lk.knn(threshold=0.01, k=2, metric=metric)._gen_similarity_pairs(all_zero))

        assert pairs == []
        assert all(0 not in pair and 1 not in pair for pair in pairs)


def test_knn_infinity_rows_match_nothing():
    """A row carrying an infinity matches nothing, under both metrics.

    The 0.0 fill leaves infinities in place. An inf row's norm is inf
    (not 0, so a zero-norm mask misses it); under cosine it normalizes
    to an exact zero vector, which sits at d=1, s=0.5 from every unit
    vector and would silently match anything below a 0.5 threshold —
    diverging from `lk.cosine`, whose 0.0 fill makes inf rows match
    nothing. Non-finite rows are excluded from the tree like zero-norm
    rows.
    """
    with_inf = pa.table({"x": [1.0, 1.0, float("inf")], "y": [0.0, 1.0, 1.0]})

    for metric in ("euclidean", "cosine"):
        pairs = list(lk.knn(threshold=0.4, k=2, metric=metric)._gen_similarity_pairs(with_inf))

        assert pairs == [(0, 1)]  # the value pair; the inf row joins nothing


def test_knn_nulls_filled_like_cosine():
    """A missing member is filled with 0.0, mirroring cosine.

    Rows [1, 1] and [None, 1] compare as [1, 1] vs [0, 1] after the
    fill: euclidean distance 1, similarity 1/(1+1) = 0.5, matched at 0.4
    and not at 0.51. Cosine on the same fill: rows normalize to
    [1/sqrt(2), 1/sqrt(2)] and [0, 1]; cos = 1/sqrt(2) ~ 0.707, matched
    at 0.4 and not at 0.8.
    """
    table = pa.table({"x": [1.0, None], "y": [1.0, 1.0]})

    pairs = list(lk.knn(threshold=0.4, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]

    pairs = list(lk.knn(threshold=0.51, k=1)._gen_similarity_pairs(table))

    assert pairs == []

    pairs = list(lk.knn(threshold=0.4, k=1, metric="cosine")._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]

    pairs = list(lk.knn(threshold=0.8, k=1, metric="cosine")._gen_similarity_pairs(table))

    assert pairs == []


def test_knn_query_called_once_per_frame(monkeypatch):
    """The mechanism is one vectorized query, not a per-pair scan.

    `cKDTree.query` runs exactly once for the whole frame, with
    `k = self._k + 1`. This is the no-all-pairs-scan structural test,
    mirroring WI049's extract spy. `cKDTree` is immutable, so the spy
    patches the module-level name `liken.dedupers.knn` binds, the same
    seam as the edit_distance cdist spy.
    """
    import liken.dedupers.knn as knn_module

    real_tree = knn_module.cKDTree
    calls = []

    class SpyTree(real_tree):
        def query(self, x, k=1, **kwargs):
            calls.append({"k": k})
            return super().query(x, k=k, **kwargs)

    monkeypatch.setattr(knn_module, "cKDTree", SpyTree)

    table = pa.table({"x": [1.0, 2.0, 3.0, None]})

    pairs = sorted(lk.knn(threshold=0.1, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2)]
    assert len(calls) == 1
    assert calls[0]["k"] == 2  # k + 1


def test_knn_single_column_table():
    """One numeric column through the compound machinery works (1-D).

    The research's noted overlap with `within`: coexistence, not
    deduplication.
    """
    table = pa.table({"x": [1.0, 2.0, 3.0]})

    pairs = sorted(lk.knn(threshold=0.3, k=1)._gen_similarity_pairs(table))

    assert pairs == [(0, 1), (1, 2)]


def test_knn_transitivity_through_union_find(mock_df):
    """Merged pairs chain transitively through union-find.

    Points A=0, B=1, C=2, k=1, threshold 0.4: pairs (A,B) and (B,C)
    qualify at 1/(1+1) = 0.5, A-C never does (d=2, s ~ 0.33). Union-find
    still merges all three through B — the docstring's transitivity note.
    """
    mock_df.get_array = Mock(return_value=pa.table({"x": [1.0, 2.0, 3.0]}))

    uf, n = lk.knn(threshold=0.4, k=1).set_frame(mock_df).build_union_find(("x",), [])

    assert n == 3
    assert uf[0] == uf[1] == uf[2]


##################
# edit_distance #
##################


def test_edit_distance_default_max_distance_is_two():
    deduper = lk.edit_distance()

    assert deduper._max_distance == 2


def test_edit_distance_matches_pairs_at_inclusive_bound():
    """Values exactly max_distance edits apart are still matched."""
    deduper = EditDistance(max_distance=2)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcde", "abcde", "abcxy"])))

    assert (0, 2) in pairs


def test_edit_distance_rejects_pairs_just_above_bound():
    """One edit beyond the bound is not matched."""
    deduper = EditDistance(max_distance=2)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcde", "abcde", "abcxyz"])))

    # only the identical pair matches; "abcde"/"abcxyz" are 3 edits apart
    assert pairs == [(0, 1)]


def test_edit_distance_rejects_long_values_with_many_edits():
    """The absolute bound holds regardless of string length."""
    deduper = EditDistance(max_distance=2)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["a" * 20 + "b", "a" * 20 + "b", "c" * 20 + "b"])))

    # only the identical pair matches; the third value is 20 edits away
    assert pairs == [(0, 1)]


def test_edit_distance_negative_max_distance_raises_value_error():
    with pytest.raises(InvalidArgumentValueError, match="`max_distance` must be an integer in the range \\[0, inf\\)"):
        lk.edit_distance(max_distance=-1)


@pytest.mark.parametrize("value", [1.5, "2", True, np.int64(2)])
def test_edit_distance_non_integer_max_distance_raises_type_error(value):
    with pytest.raises(InvalidArgumentTypeError, match="`max_distance` must be an integer, got"):
        lk.edit_distance(max_distance=value)


def test_edit_distance_default_metric_is_levenshtein():
    assert lk.edit_distance()._metric == "levenshtein"


@pytest.mark.parametrize("metric", ["damerau_levenshtein", "jaro_winkler", "indel", "hamming"])
def test_edit_distance_accepts_each_non_default_metric(metric):
    assert lk.edit_distance(max_distance=0.5, metric=metric)._metric == metric


def test_edit_distance_unknown_metric_raises_value_error():
    with pytest.raises(InvalidArgumentValueError, match="`metric` must be one of"):
        lk.edit_distance(metric="levenshteinn")


def test_edit_distance_unknown_metric_error_names_all_options():
    with pytest.raises(InvalidArgumentValueError) as exc_info:
        lk.edit_distance(metric="nope")

    message = str(exc_info.value)
    for name in ("levenshtein", "damerau_levenshtein", "jaro_winkler", "indel", "hamming"):
        assert name in message


def test_edit_distance_non_str_metric_raises_type_error():
    with pytest.raises(InvalidArgumentTypeError, match="`metric` must be of type 'str', got 'int'"):
        lk.edit_distance(metric=1)


def test_edit_distance_normalized_metric_rejects_default_max_distance():
    """The default absolute bound 2 is out of range for the normalized contract."""
    with pytest.raises(InvalidArgumentValueError, match="`max_distance` must be a real number in the range \\[0, 1\\]"):
        lk.edit_distance(metric="jaro_winkler")


def test_edit_distance_normalized_metric_accepts_float_in_range():
    assert lk.edit_distance(max_distance=0.25, metric="indel")._max_distance == 0.25


def test_edit_distance_normalized_metric_accepts_integer_zero_and_one():
    """Ints 0 and 1 sit on the inclusive bounds and are real numbers."""
    assert lk.edit_distance(max_distance=0, metric="hamming")._max_distance == 0
    assert lk.edit_distance(max_distance=1, metric="hamming")._max_distance == 1


@pytest.mark.parametrize("value", [1.5, -0.1, True, float("nan"), "0.5", np.float32(0.5)])
def test_edit_distance_normalized_metric_rejects_bad_max_distance(value):
    with pytest.raises((InvalidArgumentTypeError, InvalidArgumentValueError)):
        lk.edit_distance(max_distance=value, metric="indel")


def test_edit_distance_levenshtein_rejects_float_max_distance():
    """The absolute edit-count contract keeps its integer-only typing."""
    with pytest.raises(InvalidArgumentTypeError, match="`max_distance` must be an integer"):
        lk.edit_distance(max_distance=0.5)


#####################
# metric behaviour  #
#####################


def test_edit_distance_damerau_levenshtein_matches_transposition():
    """Damerau-Levenshtein counts a transposition as one edit: nd('abcd','abdc') = 0.25."""
    deduper = EditDistance(max_distance=0.25, metric="damerau_levenshtein")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcd", "abdc"])))

    assert pairs == [(0, 1)]

    # the same shape under plain levenshtein is 2 edits: rejected at 1, matched at 2
    assert list(lk.edit_distance(max_distance=1)._gen_similarity_pairs(pa.array(["abcd", "abdc"]))) == []
    assert list(lk.edit_distance(max_distance=2)._gen_similarity_pairs(pa.array(["abcd", "abdc"]))) == [(0, 1)]


def test_edit_distance_jaro_winkler_weights_shared_prefixes():
    """A same-shape single-edit pair scores closer when it shares a prefix.

    nd('abcdx','abcdy') = 0.08 with the shared prefix; nd('abcdx','xbcda')
    = 0.267 without it. A bound between the two separates them.
    """
    deduper = EditDistance(max_distance=0.15, metric="jaro_winkler")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcdx", "abcdy", "xbcda"])))

    assert pairs == [(0, 1)]


def test_edit_distance_indel_matches_pure_insertion():
    """Indel nd('abc','abcd') = 0.25; a substitution would cost double."""
    deduper = EditDistance(max_distance=0.25, metric="indel")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abc", "abcd"])))

    assert pairs == [(0, 1)]


def test_edit_distance_hamming_matches_substitution_pair():
    """Hamming nd('karolin','kerstin') = 3/7 ≈ 0.4286."""
    deduper = EditDistance(max_distance=0.45, metric="hamming")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["karolin", "kerstin"])))

    assert pairs == [(0, 1)]


def test_edit_distance_hamming_pads_unequal_lengths():
    """Hamming pads unequal-length values through cdist (pad=True default).

    Verified against rapidfuzz 3.14.3: nd('abc','abcde') = 0.4. Documented
    behaviour, pinned here as documentation-by-test.
    """
    deduper = EditDistance(max_distance=0.4, metric="hamming")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abc", "abcde"])))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize("metric", ["damerau_levenshtein", "indel", "hamming"])
def test_edit_distance_normalized_metric_inclusive_boundary(metric):
    """A pair exactly at `max_distance` matches; one beyond the tolerance does not.

    Exact-representable bounds (0.25 on len-4 strings) so the at-bound
    expectation does not depend on the drift guard. A bound one step below
    rejects: 0.25 exceeds it by more than the guard.
    """
    at_bound = EditDistance(max_distance=0.25, metric=metric)
    below_bound = EditDistance(max_distance=0.24, metric=metric)

    values = ["abcd", "abce"]  # one substitution over length 4: nd = 0.25

    assert list(at_bound._gen_similarity_pairs(pa.array(values))) == [(0, 1)]
    assert list(below_bound._gen_similarity_pairs(pa.array(values))) == []


def test_edit_distance_jaro_winkler_rejects_pair_above_bound():
    """A pair whose nd is clearly above the bound never matches."""
    deduper = EditDistance(max_distance=0.25, metric="jaro_winkler")

    # nd('abcdx','xbcda') = 0.2667, no shared prefix
    assert list(deduper._gen_similarity_pairs(pa.array(["abcdx", "xbcda"]))) == []


def test_edit_distance_jaro_winkler_matches_at_stubbed_exact_bound(monkeypatch):
    """A computed distance exactly at `max_distance` matches (guard direction).

    rapidfuzz's JW normalized distance can drift below the true value, so a
    real boundary pair is admitted from below; the stub pins the at-bound
    admission itself.
    """
    deduper = EditDistance(max_distance=0.11666666666666667, metric="jaro_winkler")
    monkeypatch.setattr(
        "liken.dedupers.edit_distance.process.cdist",
        lambda queries, choices, **kwargs: [[0.11666666666666667]],
    )

    assert list(deduper._gen_similarity_pairs(pa.array(["abcd", "abce"]))) == [(0, 1)]


@pytest.mark.parametrize("metric", ["levenshtein", "damerau_levenshtein", "jaro_winkler", "indel", "hamming"])
def test_edit_distance_nulls_pair_with_nulls_only_for_every_metric(metric):
    """Missing values pair only with each other, under every metric."""
    max_distance = 2 if metric == "levenshtein" else 0.5
    deduper = EditDistance(max_distance=max_distance, metric=metric)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["n", None, None, "na"])))

    assert pairs == [(0, 3), (1, 2)]


@pytest.mark.parametrize(
    "max_distance",
    [i / 100 for i in range(1, 100)],
    ids=[f"{i / 100:.2f}" for i in range(1, 100)],
)
def test_edit_distance_normalized_metric_matches_exactly_at_every_two_decimal_bound(max_distance, monkeypatch):
    """A pair whose distance is exactly `max_distance` matches, for every two-decimal bound.

    The stubbed scorer returns the bound exactly, bit-identical to
    `max_distance`, so the comparison is exact for all 99 bounds: the
    at-bound pair must match with no drift in play.
    """
    exact_distance = float(Decimal(str(max_distance)))
    deduper = EditDistance(max_distance=max_distance, metric="indel")
    monkeypatch.setattr(
        "liken.dedupers.edit_distance.process.cdist",
        lambda queries, choices, **kwargs: [[exact_distance]],
    )

    pairs = list(deduper._gen_similarity_pairs(pa.array(["x", "y"])))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize("drift", [5e-10, 2e-9], ids=["within_tolerance", "beyond_tolerance"])
def test_edit_distance_normalized_metric_tolerance_guard(drift, monkeypatch):
    """A distance within the tolerance above `max_distance` matches; beyond it does not.

    Pins the guard's widening behaviour itself: `max_distance + 5e-10`
    (inside `_BOUNDARY_TOLERANCE` = 1e-9) is admitted,
    `max_distance + 2e-9` is rejected.
    """
    deduper = EditDistance(max_distance=0.5, metric="indel")
    monkeypatch.setattr(
        "liken.dedupers.edit_distance.process.cdist",
        lambda queries, choices, **kwargs: [[0.5 + drift]],
    )

    pairs = list(deduper._gen_similarity_pairs(pa.array(["x", "y"])))

    assert pairs == ([(0, 1)] if drift == 5e-10 else [])


def test_edit_distance_normalized_metric_does_not_match_above_the_bound(monkeypatch):
    """A distance more than the tolerance above the boundary never matches."""
    deduper = EditDistance(max_distance=0.5, metric="indel")
    monkeypatch.setattr(
        "liken.dedupers.edit_distance.process.cdist",
        lambda queries, choices, **kwargs: [[0.51]],
    )

    assert list(deduper._gen_similarity_pairs(pa.array(["x", "y"]))) == []


def test_edit_distance_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError):
        EditDistance().validate(("a", "b"))


def test_edit_distance_nulls_pair_with_nulls_only():
    """Missing values pair only with each other; a null never pairs with a value.

    Recomputed from the null contract, replacing the placeholder-guard test:
    nulls never reach the scorer; Levenshtein("n", "na") is 1, so the two
    values pair at max_distance=2 as ordinary values.
    """
    deduper = EditDistance(max_distance=2)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["n", None, None, "na"])))

    assert pairs == [(0, 3), (1, 2)]


def test_edit_distance_literal_na_is_an_ordinary_value():
    """Literal "na" values pair as values; a lone null pairs with nothing.

    Recomputed from the null contract: Levenshtein("na", "") is 2, so the
    empty string pairs with both "na" literals at max_distance=2.
    """
    deduper = EditDistance(max_distance=2)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["na", None, "na", ""])))

    assert pairs == [(0, 2), (0, 3), (2, 3)]


#############
# within    #
#############


def test_within_requires_tol():
    """`tol` has no default: a tolerance is data-scale-dependent."""
    with pytest.raises(TypeError):
        lk.within()


@pytest.mark.parametrize(
    "tol",
    [None, "0.5", [1], True],
    ids=["none", "str", "list", "bool"],
)
def test_within_rejects_non_number_tol(tol):
    with pytest.raises(InvalidArgumentTypeError, match="`tol` must be a real number"):
        lk.within(tol=tol)


def test_within_rejects_huge_integer_tol():
    """An int too large for a float is refused at construction, not at
    execution with a raw OverflowError from the window arithmetic."""
    with pytest.raises(InvalidArgumentValueError, match="`tol` must be a finite number"):
        lk.within(tol=10**400)


@pytest.mark.parametrize(
    "tol",
    [-1, -0.5, float("nan"), float("inf"), float("-inf")],
    ids=["negative", "negative-float", "nan", "inf", "neg-inf"],
)
def test_within_rejects_bad_tol_value(tol):
    with pytest.raises(InvalidArgumentValueError):
        lk.within(tol=tol)


@pytest.mark.parametrize(
    "tol",
    [0, 0.5, 100],
    ids=["zero-int", "float", "int"],
)
def test_within_accepts_valid_tol(tol):
    assert lk.within(tol=tol)._tol == tol


@pytest.mark.parametrize(
    "values, tol, expected",
    [
        # close pair within tolerance
        ([10.0, 10.5, 20.0], 0.6, [(0, 1)]),
        # three-way chain: 1.0-1.01 and 1.01-1.02 match, 1.0-1.02 does not
        ([1.0, 1.01, 1.02], 0.01, [(0, 1), (1, 2)]),
        # far apart: nothing matches
        ([1.0, 5.0, 9.0], 1.0, []),
        # unsorted input still pairs correctly
        ([30.0, 10.1, 20.0, 10.0], 0.2, [(1, 3)]),
        # duplicates match each other
        ([7.0, 7.0, 7.0], 0.0, [(0, 1), (0, 2), (1, 2)]),
    ],
)
def test_within_matches_pairs_within_tolerance(values, tol, expected):
    pairs = sorted(lk.within(tol=tol)._gen_similarity_pairs(pa.array(values)))

    assert pairs == expected


@pytest.mark.parametrize(
    "a, b, tol",
    [
        (1.0, 1.01, 0.01),
        (100.0, 100.4, 0.4),
    ],
)
def test_within_matches_inclusively_at_the_bound(a, b, tol):
    pairs = list(lk.within(tol=tol)._gen_similarity_pairs(pa.array([a, b])))

    assert pairs == [(0, 1)]


def test_within_rejects_pair_just_above_bound():
    pairs = list(lk.within(tol=0.01)._gen_similarity_pairs(pa.array([1.0, 1.011])))

    assert pairs == []


def test_within_boundary_guard_admits_rounding_drift():
    """`1.02 - 1.01` computes to 0.010000000000000009, above tol=0.01.

    The guard widens the bound by 1e-9 so the spec's own example matches.
    """
    pairs = list(lk.within(tol=0.01)._gen_similarity_pairs(pa.array([1.01, 1.02])))

    assert pairs == [(0, 1)]


def test_within_zero_tol_is_exact_equality():
    """`tol=0` reduces to numeric equality: only exactly equal values pair.

    No boundary guard applies at tol=0: equal values subtract to exactly
    0.0, so equality is exact, not guarded.
    """
    pairs = sorted(lk.within(tol=0)._gen_similarity_pairs(pa.array([1.0, 1.0, 1.5, 2])))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize(
    "values",
    [[1.0, None, None, 5.0], [1.0, float("nan"), None, 5.0]],
    ids=["none-nulls", "nan-and-null"],
)
def test_within_nulls_pair_with_nulls_only(values):
    """Nulls pair with nulls only; the two values are too far apart to pair."""
    pairs = sorted(lk.within(tol=0.5)._gen_similarity_pairs(pa.array(values)))

    assert pairs == [(1, 2)]


def test_within_literal_zero_is_a_value_not_missing():
    """0.0 is a value: it pairs with 0.0, not with a null."""
    pairs = sorted(lk.within(tol=0.1)._gen_similarity_pairs(pa.array([0.0, None, 0.05])))

    assert pairs == [(0, 2)]


def test_within_empty_and_single_value_arrays_yield_nothing():
    deduper = lk.within(tol=1.0)

    assert list(deduper._gen_similarity_pairs(pa.array([]))) == []
    assert list(deduper._gen_similarity_pairs(pa.array([1.0]))) == []


def test_within_infinite_values_never_match():
    """Two infinities subtract to NaN, which matches nothing."""
    pairs = list(lk.within(tol=1.0)._gen_similarity_pairs(pa.array([float("inf"), float("inf")])))

    assert pairs == []


def test_within_large_integers_survive_float_conversion():
    """Integers beyond float53 precision collapse to the same float and pair.

    Python ints keep full precision; converting to float before comparison
    would merge 2**53 and 2**53 + 1. The pair decision runs on the raw
    values.
    """
    big = 2**53
    pairs = list(lk.within(tol=0)._gen_similarity_pairs(pa.array([big, big + 1], type=pa.int64())))

    assert pairs == []


def test_within_pair_emission_is_deterministic():
    """Repeated runs over the same input emit identical pair lists.

    Multi-pair input: emission order is window-scan order over the sorted
    values, not globally sorted; only run-to-run stability is claimed.
    """
    deduper = lk.within(tol=0.5)

    first = list(deduper._gen_similarity_pairs(pa.array([3.0, 1.2, 1.0, 1.1])))
    second = list(deduper._gen_similarity_pairs(pa.array([3.0, 1.2, 1.0, 1.1])))

    assert first == [(2, 3), (1, 2), (1, 3)]
    assert first == second


@pytest.mark.parametrize(
    "array",
    [
        pa.array(["a", "b"]),
        pa.array([True, False]),
        pa.array([date(2024, 1, 1), date(2024, 1, 2)]),
    ],
    ids=["strings", "booleans", "dates"],
)
def test_within_rejects_non_numeric_arrays_at_execution(array):
    with pytest.raises(InvalidArgumentTypeError, match="numeric"):
        list(lk.within(tol=1.0)._gen_similarity_pairs(array))


@pytest.mark.parametrize(
    "array",
    [
        pa.array([1, 2, 3]),
        pa.array([Decimal("1.5"), Decimal("2.5")]),
        pa.array([1.0, 2.0]),
    ],
    ids=["integers", "decimals", "floats"],
)
def test_within_accepts_numeric_arrays(array):
    assert isinstance(list(lk.within(tol=1.0)._gen_similarity_pairs(array)), list)


def test_within_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError):
        lk.within(tol=1.0).validate(("a", "b"))


def test_within_str_renders_tol():
    assert "within(tol=0.5)" in str(lk.within(tol=0.5))


################################
# tfidf and lsh null handling #
###############################


@pytest.mark.parametrize("ngram", [1, 2, 3])
def test_tfidf_null_never_pairs_with_value(ngram):
    """A null pairs only with another null, at any ngram.

    "nathan" and "nadia" share no character 3-grams, so at the default ngram
    the null-null pair is the only pair; the assertion allows value pairs but
    forbids any pair touching a null other than the null-null pair.
    """
    array = pa.array(["nathan", None, "nadia", None])

    pairs = sorted(lk.tfidf(ngram=ngram)._gen_similarity_pairs(array))

    assert (1, 3) in pairs
    assert all(1 not in pair and 3 not in pair for pair in pairs if pair != (1, 3))


def test_tfidf_all_null_input_yields_star_pairs():
    """All-null input yields star-shaped null pairs without touching the vectoriser."""
    array = pa.array([None, None, None])

    assert list(lk.tfidf()._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


@pytest.mark.parametrize("ngram", [1, 2, 3])
def test_lsh_null_never_pairs_with_value(ngram):
    """A null pairs only with another null, at any ngram."""
    array = pa.array(["nathan", None, "nadia", None])

    pairs = sorted(lk.lsh(ngram=ngram)._gen_similarity_pairs(array))

    assert (1, 3) in pairs
    assert all(1 not in pair and 3 not in pair for pair in pairs if pair != (1, 3))


def test_lsh_all_null_input_yields_star_pairs():
    """All-null input yields star-shaped null pairs without touching datasketch."""
    array = pa.array([None, None, None])

    assert list(lk.lsh()._gen_similarity_pairs(array)) == [(0, 1), (0, 2)]


####################################
# str factory construction checks  #
####################################


@pytest.mark.parametrize(
    "factory",
    [lk.str_contains, lk.str_startswith, lk.str_endswith],
    ids=["str_contains", "str_startswith", "str_endswith"],
)
def test_str_factories_reject_empty_pattern(factory):
    """An empty pattern matches every value and silently collapses the column."""
    with pytest.raises(InvalidArgumentValueError, match="`pattern` must be a non-empty string, got ''"):
        factory(pattern="")


@pytest.mark.parametrize(
    "factory, pattern",
    [
        (lk.str_contains, 123),
        (lk.str_startswith, ["cal"]),
        (lk.str_endswith, None),
    ],
    ids=["str_contains", "str_startswith", "str_endswith"],
)
def test_str_factories_reject_non_str_pattern(factory, pattern):
    with pytest.raises(InvalidArgumentTypeError, match="`pattern` must be of type 'str', got"):
        factory(pattern=pattern)


@pytest.mark.parametrize("kwargs", [{"min_len": -1}, {"max_len": -1}])
def test_str_len_rejects_negative_bounds(kwargs):
    """A negative length bound is refused at construction; it behaves as 0."""
    with pytest.raises(InvalidArgumentValueError, match="must be an integer in the range \\[0, inf\\), got -1"):
        lk.str_len(**kwargs)


################################
# eager column validation     #
###############################


def test_col_chaining_validates_columns_eagerly():
    """a chained deduper's column validation runs at construction."""
    with pytest.raises(InvalidArgumentTypeError):
        lk.col(("a", "b")).str_contains(pattern="x")


def test_col_chaining_eagerly_rejects_non_str_tuple_member():
    with pytest.raises(InvalidArgumentTypeError, match="`columns` members must be of type 'str'"):
        lk.col(("a", 1)).exact()


def test_compound_column_validation_rejects_non_str_member():
    with pytest.raises(InvalidArgumentTypeError, match="`columns` members must be of type 'str'"):
        Jaccard().validate(("a", 1))


def test_exact_validation_accepts_str_and_tuple_of_str():
    """U29: exact validates 'str or tuple-of-str' instead of opting out."""
    lk.exact().validate("a")
    lk.exact().validate(("a", "b"))
    with pytest.raises(InvalidArgumentTypeError, match="`columns` must be of type 'str' or 'tuple', got 'int'"):
        lk.exact().validate(123)


###############################
# isin construction validation #
###############################


@pytest.mark.parametrize(
    "values",
    ["london", 42, {"london": 1}, object()],
    ids=["str", "int", "dict", "object"],
)
def test_isin_rejects_non_collection_values(values):
    """`values` must be a list, tuple, set or frozenset; a str is an iterable
    and was silently treated as a character collection."""
    with pytest.raises(
        InvalidArgumentTypeError,
        match="`values` must be of type 'list' or 'tuple' or 'set' or 'frozenset', got",
    ):
        lk.isin(values=values)


@pytest.mark.parametrize(
    "values",
    [["london"], ("london",), {"london"}, frozenset({"london"})],
    ids=["list", "tuple", "set", "frozenset"],
)
def test_isin_materialises_values_to_tuple(values):
    """Values are materialised at construction, so a lazy iterable cannot be
    consumed by the first match."""
    deduper = lk.isin(values=values)

    assert deduper._values == ("london",)


###############################
# lsh construction validation #
###############################


@pytest.mark.parametrize("ngram", [0, -1])
def test_lsh_rejects_non_positive_ngram(ngram):
    """`ngram` below 1 silently collapses the column at execution; refused at construction."""
    with pytest.raises(InvalidArgumentValueError, match="`ngram` must be an integer in the range \\[1, inf\\), got"):
        lk.lsh(ngram=ngram)


@pytest.mark.parametrize("ngram", ["3", 3.0, True, None])
def test_lsh_rejects_non_integer_ngram(ngram):
    with pytest.raises(InvalidArgumentTypeError, match="`ngram` must be an integer, got"):
        lk.lsh(ngram=ngram)


@pytest.mark.parametrize("num_perm", [0, -5])
def test_lsh_rejects_non_positive_num_perm(num_perm):
    with pytest.raises(InvalidArgumentValueError, match="`num_perm` must be an integer in the range \\[1, inf\\), got"):
        lk.lsh(num_perm=num_perm)


@pytest.mark.parametrize("num_perm", [128.0, "128", True, None])
def test_lsh_rejects_non_integer_num_perm(num_perm):
    with pytest.raises(InvalidArgumentTypeError, match="`num_perm` must be an integer, got"):
        lk.lsh(num_perm=num_perm)


def test_lsh_accepts_valid_ngram_and_num_perm():
    lk.lsh(ngram=3, num_perm=128)


###################################
# tfidf construction validation  #
##################################


@pytest.mark.parametrize("topn", [0, -1])
def test_tfidf_rejects_non_positive_topn(topn):
    """`topn` below 1 is refused at construction: an unvalidated topn reaches
    the native library and can abort the process."""
    with pytest.raises(InvalidArgumentValueError, match="`topn` must be an integer in the range \\[1, inf\\), got"):
        lk.tfidf(topn=topn)


@pytest.mark.parametrize("topn", [2.0, "2", True, None])
def test_tfidf_rejects_non_integer_topn(topn):
    with pytest.raises(InvalidArgumentTypeError, match="`topn` must be an integer, got"):
        lk.tfidf(topn=topn)


@pytest.mark.parametrize("ngram", [0, -1])
def test_tfidf_rejects_non_positive_ngram(ngram):
    with pytest.raises(InvalidArgumentValueError, match="`ngram` must be an integer in the range \\[1, inf\\), got"):
        lk.tfidf(ngram=ngram)


@pytest.mark.parametrize("ngram", ["3", 3.0, True, None])
def test_tfidf_rejects_non_integer_ngram(ngram):
    with pytest.raises(InvalidArgumentTypeError, match="`ngram` must be an integer or a 2-tuple of integers, got"):
        lk.tfidf(ngram=ngram)


@pytest.mark.parametrize("ngram", [(2, 1), (0, 2), (1, 0), (2, 2.0), (2, "3")])
def test_tfidf_rejects_invalid_ngram_tuple(ngram):
    """A 2-tuple ngram range must be integers with 1 <= low <= high."""
    with pytest.raises((InvalidArgumentTypeError, InvalidArgumentValueError)):
        lk.tfidf(ngram=ngram)


@pytest.mark.parametrize("ngram", [(2,), (1, 2, 3), []])
def test_tfidf_rejects_wrong_shape_ngram_tuple(ngram):
    with pytest.raises(InvalidArgumentTypeError, match="`ngram` must be an integer or a 2-tuple of integers, got"):
        lk.tfidf(ngram=ngram)


def test_tfidf_accepts_valid_ngram_and_topn():
    lk.tfidf(ngram=3, topn=2)
    lk.tfidf(ngram=(1, 2), topn=1)


##############
# num_range #
##############


def test_num_range_rejects_both_bounds_none():
    """A fully open range matches every non-missing value; refused at construction."""
    with pytest.raises(ValueError):
        lk.num_range(min=None, max=None)


def test_num_range_rejects_min_above_max():
    """An empty interval is refused at construction, like both-None."""
    with pytest.raises(ValueError):
        lk.num_range(min=40_000, max=30_000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": True},
        {"max": True},
        {"min": False},
        {"min": "500"},
        {"max": "500"},
    ],
    ids=["min-bool", "max-bool", "min-false", "min-str", "max-str"],
)
def test_num_range_rejects_non_number_bounds(kwargs):
    """A bound that is not a real number — bool, str — is refused at construction."""
    with pytest.raises(InvalidArgumentTypeError, match="must be a real number, got"):
        lk.num_range(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": float("nan")},
        {"max": float("nan")},
        {"min": float("nan"), "max": 5},
    ],
    ids=["min-nan", "max-nan", "min-nan-with-max"],
)
def test_num_range_rejects_nan_bounds(kwargs):
    """A NaN bound is refused at construction as an invalid value."""
    with pytest.raises(InvalidArgumentValueError, match="must be a real number, got NaN"):
        lk.num_range(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min": 1, "max": 2},
        {"min": 1.5},
        {"max": 1.5},
        {"min": None, "max": 2},
        {"min": 1, "max": None},
    ],
    ids=["ints", "min-float", "max-float", "min-none", "max-none"],
)
def test_num_range_allows_real_or_none_bounds(kwargs):
    """An int or float bound is a real number; `None` means unbounded."""
    lk.num_range(**kwargs)


def test_num_range_min_equals_max_is_allowed():
    """A one-value interval is a valid, non-empty range."""
    lk.num_range(min=5, max=5)


@pytest.mark.parametrize(
    "kwargs, values, expected_pairs",
    [
        # band: only 3, 4 and 5 sit inside; pairs are star-shaped from the
        # first matched index (predicate semantics: one collapsed group)
        ({"min": 3, "max": 5}, [1, 2, 3, 4, 5, 6], [(2, 3), (2, 4)]),
        # inclusive endpoints: 1 and 5 sit at the bounds and must match
        ({"min": 1, "max": 5}, [1, 2, 5], [(0, 1), (0, 2)]),
        # min-only: everything >= 30_000
        ({"min": 30_000}, [29_999, 30_000, 30_001], [(1, 2)]),
        # max-only: everything <= 40_000
        ({"max": 40_000}, [39_999, 40_000, 40_001], [(0, 1)]),
    ],
    ids=["band", "inclusive-endpoints", "min-only", "max-only"],
)
def test_num_range_matches_values_inside_inclusive_bounds(kwargs, values, expected_pairs):
    """A value matches iff min <= v <= max, each bound applied only when given."""
    deduper = lk.num_range(**kwargs)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == expected_pairs


def test_num_range_empty_match_set_yields_no_pairs():
    """No value in range: no pairs, no crash."""
    deduper = lk.num_range(min=100, max=200)

    assert list(deduper._gen_similarity_pairs(pa.array([1, 2, 3]))) == []


def test_num_range_null_and_nan_never_match_positive():
    """On the positive path, a missing value matches nothing."""
    deduper = lk.num_range(min=1, max=5)

    pairs = list(deduper._gen_similarity_pairs(pa.array([1.0, None, float("nan"), 5.0])))

    assert pairs == [(0, 3)]


def test_num_range_null_and_nan_never_match_negated():
    """On the negated path, a missing value matches nothing.

    The Arrow comparison kernels return False at a float NaN, so a bare
    `pc.invert` of the positive mask would group NaN rows. The deduper's
    mask must null out missing positions before the inversion.
    """
    deduper = ~lk.num_range(min=1, max=5)

    pairs = list(deduper._gen_similarity_pairs(pa.array([0.5, None, float("nan"), 9.0])))

    assert pairs == [(0, 3)]


def test_num_range_fallback_never_matches_missing():
    """The Python fallback path also never matches a missing value."""
    deduper = lk.num_range(min=1, max=5)

    assert deduper._matches(None) is False
    assert deduper._matches(float("nan")) is False
    assert deduper._matches(3) is True


def test_num_range_negated_fallback_never_matches_missing():
    """The negated Python fallback also never matches a missing value."""
    deduper = ~lk.num_range(min=1, max=5)

    assert deduper._matches(None) is False
    assert deduper._matches(float("nan")) is False
    assert deduper._matches(3) is False
    assert deduper._matches(50) is True


def test_num_range_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError):
        lk.num_range(min=1).validate(("a", "b"))


def test_num_range_str_renders_name_and_bounds():
    representation = str(lk.num_range(min=1, max=5))

    assert representation == "num_range(min=1, max=5)"


##############
#  str_len  #
##############


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_len": True},
        {"max_len": True},
        {"min_len": False},
        {"min_len": "3"},
        {"max_len": "3"},
        {"min_len": 3.0},
        {"max_len": 3.0},
        {"min_len": None},
    ],
    ids=["min-bool", "max-bool", "min-false", "min-str", "max-str", "min-float", "max-float", "min-none"],
)
def test_str_len_rejects_non_integer_bounds(kwargs):
    """A bound that is not an integer — bool, str, float, `None` — is refused at construction."""
    with pytest.raises(InvalidArgumentTypeError, match="must be an integer, got"):
        lk.str_len(**kwargs)


def test_str_len_rejects_inverted_bounds():
    """An empty interval is refused at construction, like num_range."""
    with pytest.raises(ValueError):
        lk.str_len(min_len=10, max_len=9)


def test_str_len_allows_int_bounds_and_none_upper():
    """An int bound is valid; `max_len` may be `None`; `min_len == max_len` is legal."""
    lk.str_len(min_len=3, max_len=None)
    lk.str_len(min_len=3, max_len=3)


@pytest.mark.parametrize(
    "kwargs, values, expected_pairs",
    [
        # band: lengths 3, 4 and 5 sit inside; pairs are star-shaped from
        # the first matched index (predicate semantics: one collapsed group)
        ({"min_len": 3, "max_len": 5}, ["ab", "abc", "abcd", "abcde", "abcdef"], [(1, 2), (1, 3)]),
        # inclusive endpoints: lengths 3 and 5 sit at the bounds and must match
        ({"min_len": 3, "max_len": 5}, ["abc", "abcd", "abcde"], [(0, 1), (0, 2)]),
        # min-only: everything of length >= 2
        ({"min_len": 2}, ["a", "ab", "abc"], [(1, 2)]),
        # max-only: everything of length <= 2
        ({"max_len": 2}, ["a", "ab", "abc"], [(0, 1)]),
    ],
    ids=["band", "inclusive-endpoints", "min-only", "max-only"],
)
def test_str_len_matches_lengths_inside_inclusive_bounds(kwargs, values, expected_pairs):
    """A value matches iff min_len <= len(v) <= max_len, each bound applied only when given."""
    deduper = lk.str_len(**kwargs)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == expected_pairs


def test_str_len_min_equals_max_matches_exact_length():
    """min_len == max_len is the exact-length idiom, symmetric with num_range's one-value interval."""
    deduper = lk.str_len(min_len=3, max_len=3)

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["abc", "xyz", "abcd", "ab"])))

    assert pairs == [(0, 1)]


def test_str_len_empty_string_never_matches_positive():
    """The empty string is excluded whatever the bounds, even at min_len=0."""
    deduper = lk.str_len(min_len=0)

    assert list(deduper._gen_similarity_pairs(pa.array(["", "ab"]))) == []


def test_str_len_empty_string_matches_only_under_negation():
    """Under ~str_len the empty string negate-matches, like every non-matching value."""
    deduper = ~lk.str_len(min_len=0)

    assert list(deduper._gen_similarity_pairs(pa.array(["", "", "ab"]))) == [(0, 1)]


def test_str_len_mask_is_null_at_missing_positions():
    """Missing positions are null in the mask, not False.

    An explicitly nulled mask survives `pc.invert` on the negated path,
    whatever the base mask's missing-value behaviour evolves into.
    """
    deduper = lk.str_len(min_len=1, max_len=3)

    mask = deduper._vectorized_matches(pa.array(["ab", None]))

    assert mask.is_valid().to_pylist() == [True, False]


def test_str_len_null_never_matches_positive():
    """On the positive path, a missing value matches nothing."""
    deduper = lk.str_len(min_len=1, max_len=3)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["ab", None, "abc"])))

    assert pairs == [(0, 2)]


def test_str_len_null_never_matches_negated():
    """On the negated path, a missing value matches nothing."""
    deduper = ~lk.str_len(min_len=1, max_len=3)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["abcd", None, ""])))

    assert pairs == [(0, 2)]


def test_str_len_fallback_never_matches_missing_or_empty():
    """The Python fallback path: a missing value or the empty string never matches."""
    deduper = lk.str_len(min_len=3, max_len=5)

    assert deduper._matches(None) is False
    assert deduper._matches("") is False
    assert deduper._matches("abc") is True
    assert deduper._matches("ab") is False
    assert deduper._matches("abcdef") is False


def test_str_len_negated_fallback_never_matches_missing():
    """The negated Python fallback also never matches a missing value; the empty string does."""
    deduper = ~lk.str_len(min_len=3, max_len=5)

    assert deduper._matches(None) is False
    assert deduper._matches("") is True
    assert deduper._matches("abc") is False
    assert deduper._matches("ab") is True


################
#  date_range  #
################


def test_date_range_requires_format():
    """`format` has no default: the deduper never guesses a date format."""
    with pytest.raises(TypeError):
        lk.date_range()


@pytest.mark.parametrize(
    "format",
    [None, 5, ["%Y-%m-%d"], True],
    ids=["none", "int", "list", "bool"],
)
def test_date_range_rejects_non_str_format(format):
    with pytest.raises(InvalidArgumentTypeError, match="`format` must be of type 'str', got"):
        lk.date_range(format=format)


def test_date_range_rejects_empty_format():
    with pytest.raises(InvalidArgumentValueError, match="`format` must be a non-empty string"):
        lk.date_range(format="")


@pytest.mark.parametrize(
    "window",
    [1.0, "1", True, None],
    ids=["float", "str", "bool", "none"],
)
def test_date_range_rejects_non_integer_window(window):
    with pytest.raises(InvalidArgumentTypeError, match="`window` must be an integer"):
        lk.date_range(format="%Y-%m-%d", window=window)


def test_date_range_rejects_negative_window():
    with pytest.raises(InvalidArgumentValueError, match="`window` must be an integer in the range"):
        lk.date_range(format="%Y-%m-%d", window=-1)


@pytest.mark.parametrize(
    "window",
    [0, 1, 30],
    ids=["zero", "one", "thirty"],
)
def test_date_range_accepts_valid_window(window):
    assert lk.date_range(format="%Y-%m-%d", window=window)._window == window


def test_date_range_default_window_is_zero():
    assert lk.date_range(format="%Y-%m-%d")._window == 0


def test_date_range_boundary_crossing_bucket_merge():
    """Jan 5 and Jan 7 at window=2 merge even though plain date-key
    bucketing would separate them; this pins the window+1 cell scheme."""
    deduper = lk.date_range(format="%Y-%m-%d", window=2)
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["2024-01-05", "2024-01-07"])))

    assert pairs == [(0, 1)]


def test_date_range_chain_at_window_one_emits_adjacent_pairs_only():
    """Jan 5, 6, 7 at window=1: adjacent pairs match, Jan 5-Jan 7 does not."""
    deduper = lk.date_range(format="%Y-%m-%d", window=1)
    deduper.validate("date")

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(["2024-01-05", "2024-01-06", "2024-01-07"])))

    assert pairs == [(0, 1), (1, 2)]


def test_date_range_dense_two_dates_at_window_zero():
    """Two dates 1 day apart at window=0 emit only the within-day stars.

    The neighbour join previously cross-multiplied the two cells (24.7s at
    n=30k for zero cross-cell pairs); the two-pointer scan visits each
    side once. Same two dates one day apart at window=1 merge everything.
    """
    values = ["2024-01-05"] * 3 + ["2024-01-06"] * 3

    zero = lk.date_range(format="%Y-%m-%d")
    zero.validate("date")
    assert sorted(zero._gen_similarity_pairs(pa.array(values))) == [(0, 1), (0, 2), (3, 4), (3, 5)]

    one = lk.date_range(format="%Y-%m-%d", window=1)
    one.validate("date")
    pairs = sorted(one._gen_similarity_pairs(pa.array(values)))

    # The within-day pairs are star-shaped (predicate convention); the
    # cross-day join emits every matching pair. Grouping: all six merge.
    assert pairs == [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
    ]


def test_date_range_dense_mixed_dates_cross_cell():
    """Dense data over several days: the two-pointer neighbour join emits
    every matching cross-cell pair; within-cell emission is star-shaped
    from the cell's first index (predicate convention). The merged groups
    equal the brute-force within-window groups; pair shape is
    cell-alignment dependent."""
    values = [f"2024-01-{day:02d}" for day in (5, 5, 6, 6, 7, 7, 9, 10)]

    deduper = lk.date_range(format="%Y-%m-%d", window=1)
    deduper.validate("date")

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (2, 5),
        (6, 7),
    ]


def test_date_range_pre_epoch_dates_match_within_window():
    """Dates before 1970-01-01 pair correctly: the day ordinal is a signed
    integer, so a floor-to-day cast keeps the arithmetic correct."""
    deduper = lk.date_range(format="%Y-%m-%d", window=1)
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["1969-12-31", "1970-01-01"])))

    assert pairs == [(0, 1)]


def test_date_range_values_reversed_order_still_pairs():
    deduper = lk.date_range(format="%Y-%m-%d", window=2)
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["2024-01-07", "2024-01-05"])))

    assert sorted(pairs) == [(0, 1)]


def test_sorted_tokens_accepts_large_string_arrays():
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["a b", "b a"], type=pa.large_string())))

    assert pairs == [(0, 1)]


def test_sorted_tokens_null_typed_array_yields_missing_pairs_only():
    """A `null`-typed array (e.g. an all-empty column) carries no tokens:
    only the missing-star pairs emit."""
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array([None, None], type=pa.null())))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize(
    "values",
    [["2024-01-05", None, None, "2024-01-06"]],
    ids=["none-nulls"],
)
def test_date_range_nulls_pair_with_nulls_only(values):
    """Nulls pair with nulls only, star-shaped; null never pairs with a date.

    The dates sit one day apart and `window=30` would merge them, so the
    only pair is the null-null pair."""
    deduper = lk.date_range(format="%Y-%m-%d", window=30)
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == [(1, 2), (0, 3)]


def test_date_range_parse_failure_names_column_and_format():
    """A value that does not parse raises, naming the column and format."""
    deduper = lk.date_range(format="%Y-%m-%d", window=0)
    deduper.validate("birth_date")

    with pytest.raises(InvalidArgumentValueError, match=r"birth_date.*%Y-%m-%d"):
        list(deduper._gen_similarity_pairs(pa.array(["2024-01-05", "05/01/2024"])))


def test_date_range_mixed_formats_fail_under_one_format():
    """Mixed formats are the tar pit the spec forbids solving: the second
    format fails under the first, loudly."""
    deduper = lk.date_range(format="%d/%m/%Y", window=0)
    deduper.validate("date")

    with pytest.raises(InvalidArgumentValueError, match="column 'date'"):
        list(deduper._gen_similarity_pairs(pa.array(["05/01/2024", "2024-01-05"])))


def test_date_range_rejects_non_string_arrays_at_execution():
    deduper = lk.date_range(format="%Y-%m-%d")
    deduper.validate("n")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_date_range_boolean_array_rejected_at_execution():
    deduper = lk.date_range(format="%Y-%m-%d")
    deduper.validate("flag")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([True, False])))


def test_date_range_negated_rejects_non_string_array():
    """The negated path type-checks the column too: without the check a
    non-string column fails to parse per value and `~date_range` silently
    matches every row."""
    deduper = ~lk.date_range(format="%Y-%m-%d")
    deduper.validate("n")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_date_range_empty_array_yields_no_pairs():
    deduper = lk.date_range(format="%Y-%m-%d")
    deduper.validate("date")

    assert list(deduper._gen_similarity_pairs(pa.array([], type=pa.string()))) == []


def test_date_range_parse_failure_on_all_values():
    """Every value failing to parse raises, not silently no pairs."""
    deduper = lk.date_range(format="%Y-%m-%d")
    deduper.validate("date")

    with pytest.raises(InvalidArgumentValueError, match="column 'date'"):
        list(deduper._gen_similarity_pairs(pa.array(["nope", "also-nope"])))


def test_date_range_negated_groups_unparseable_values():
    """~date_range groups values that fail to parse (the coherent per-row
    reading: _matches is 'value parses under format')."""
    deduper = ~lk.date_range(format="%Y-%m-%d")
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["garbage", "also garbage", "2024-01-05"])))

    assert pairs == [(0, 1)]


def test_date_range_negated_never_matches_missing():
    """A missing value matches nothing on either polarity."""
    deduper = ~lk.date_range(format="%Y-%m-%d")
    deduper.validate("date")

    pairs = list(deduper._gen_similarity_pairs(pa.array([None, "2024-01-05"])))

    assert pairs == []


def test_date_range_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError):
        lk.date_range(format="%Y-%m-%d").validate(("a", "b"))


def test_date_range_pair_emission_is_deterministic():
    deduper = lk.date_range(format="%Y-%m-%d", window=1)
    deduper.validate("date")

    values = ["2024-01-06", "2024-01-05", "2024-01-06"]

    first = list(deduper._gen_similarity_pairs(pa.array(values)))
    second = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert first == second


def test_date_range_str_renders_format_and_window():
    assert "date_range(format='%Y-%m-%d', window=1)" in str(lk.date_range(format="%Y-%m-%d", window=1))


###################
#  sorted_tokens  #
###################


@pytest.mark.parametrize(
    "values",
    [
        ["John Smith", "Smith John"],
        ["John Smith", "John Smith"],
        ["  John   Smith  ", "John Smith"],
        ["John\tSmith", "John Smith"],
        ["Mary Ann Lee", "Lee Ann Mary"],
    ],
    ids=["swap", "identical", "padded", "tab", "three-token-swap"],
)
def test_sorted_tokens_groups_word_order_swaps(values):
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == [(0, 1)]


@pytest.mark.parametrize(
    "values",
    [
        ["John Smith", "Jon Smith"],
        ["John Smith", "John Smithy"],
        ["John Smith", "John  Smithy"],
        ["Main St.", "Main, St."],
    ],
    ids=["different-word", "different-token", "same-multiset", "punctuation-placement"],
)
def test_sorted_tokens_never_groups_different_token_multisets(values):
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == []


def test_sorted_tokens_single_token_values_behave_as_exact():
    """A single token sorts to itself: matching degenerates to equality."""
    deduper = lk.sorted_tokens()
    deduper.validate("city")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["london", "london", "london2"])))

    assert pairs == [(0, 1)]


def test_sorted_tokens_transitive_grouping(mock_df):
    """Values sharing a sorted-token key merge through union-find; distinct
    keys stay apart."""
    mock_df.get_array = Mock(return_value=pa.array(["John Smith", "Smith John", "John Smith", "Peter Paul"]))

    uf, n = lk.sorted_tokens().set_frame(mock_df).build_union_find("name", [])

    assert n == 4
    assert uf[0] == uf[1] == uf[2]
    assert uf[3] != uf[0]


@pytest.mark.parametrize(
    "values",
    [
        ["John Smith", "", "Jane Doe", ""],
        ["John Smith", "   ", "Jane Doe"],
    ],
    ids=["empty", "whitespace-only"],
)
def test_sorted_tokens_zero_token_values_never_match(values):
    """An empty or whitespace-only string carries no tokens: it matches
    nothing on the positive path, not even its twin."""
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == []


def test_sorted_tokens_nulls_pair_with_nulls_only():
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(["John Smith", None, None, "Jane Doe"])))

    assert pairs == [(1, 2)]


@pytest.mark.parametrize(
    "values",
    [
        ["John Smith", "", "   ", None, "Smith John"],
        ["John Smith", "", None, ""],
    ],
    ids=["both-zero-token-classes", "two-empties"],
)
def test_sorted_tokens_negated_groups_zero_token_values(values):
    """~sorted_tokens groups zero-token values (empty and whitespace-only);
    values with tokens and nulls match nothing under negation."""
    deduper = ~lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array(values)))

    zero_token = [i for i, value in enumerate(values) if isinstance(value, str) and not value.split()]
    assert len(zero_token) >= 2
    assert all(a in zero_token and b in zero_token for a, b in pairs)
    # every pair of zero-token values pairs
    expected = {(a, b) for i, a in enumerate(zero_token) for b in zero_token[i + 1 :]}
    assert set(pairs) == expected


def test_sorted_tokens_negated_never_matches_nulls():
    deduper = ~lk.sorted_tokens()
    deduper.validate("name")

    pairs = list(deduper._gen_similarity_pairs(pa.array([None, "John Smith"])))

    assert pairs == []


def test_sorted_tokens_rejects_non_string_arrays_at_execution():
    deduper = lk.sorted_tokens()
    deduper.validate("n")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_sorted_tokens_boolean_array_rejected_at_execution():
    deduper = lk.sorted_tokens()
    deduper.validate("flag")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([True, False])))


def test_sorted_tokens_negated_rejects_non_string_array():
    """The negated path type-checks the column too: without the check a
    non-string column would fail `_matches` per value and `~sorted_tokens`
    would silently star-group every row."""
    deduper = ~lk.sorted_tokens()
    deduper.validate("n")

    with pytest.raises(InvalidArgumentTypeError, match="string"):
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_sorted_tokens_empty_array_yields_no_pairs():
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    assert list(deduper._gen_similarity_pairs(pa.array([], type=pa.string()))) == []


def test_sorted_tokens_requires_single_string_column():
    with pytest.raises(InvalidArgumentTypeError, match="`columns` must be of type 'str'"):
        lk.sorted_tokens().validate(("a", "b"))


def test_sorted_tokens_permutation_invariance(mock_df):
    """The same multiset of values in different row orders yields the same
    grouping up to relabelling (WI017 concern): sort keys are token strings
    in code-point order, no locale collation. Comparison is on partitions
    (frozensets of member indices), not on raw union-find roots."""
    values = ["John Smith", "Smith John", "Mary Ann Lee", "Lee Mary Ann", "Ann Lee Mary", "Peter Paul"]

    def grouping(order):
        mock_df.get_array = Mock(return_value=pa.array([values[i] for i in order]))
        uf, _ = lk.sorted_tokens().set_frame(mock_df).build_union_find("name", [])
        members: dict[object, set[int]] = {}
        for i in range(len(order)):
            members.setdefault(uf[i], set()).add(i)
        return {frozenset(group) for group in members.values()}

    rng = random.Random(7)
    reference = grouping(list(range(len(values))))
    for _ in range(20):
        order = list(range(len(values)))
        rng.shuffle(order)
        shuffled = grouping(order)
        # shuffle[i] holds values[order[i]]; map each shuffled group to the
        # values it holds, so the two partitions compare directly
        relabelled = {frozenset(order[i] for i in group) for group in shuffled}
        assert relabelled == reference


def test_sorted_tokens_pair_emission_is_deterministic():
    deduper = lk.sorted_tokens()
    deduper.validate("name")

    values = ["John Smith", "Smith John", "Mary Ann Lee", "Lee Mary Ann"]

    first = list(deduper._gen_similarity_pairs(pa.array(values)))
    second = list(deduper._gen_similarity_pairs(pa.array(values)))

    assert first == second


def test_sorted_tokens_str_renders_no_arguments():
    assert "sorted_tokens()" in str(lk.sorted_tokens())


##################################
# threshold boundary semantics  #
##################################


@pytest.mark.parametrize(
    "threshold",
    [i / 100 for i in range(1, 100)],
    ids=[f"{i / 100:.2f}" for i in range(1, 100)],
)
def test_fuzzy_matches_score_exactly_at_every_two_decimal_threshold(threshold, monkeypatch):
    """A pair whose score is exactly `100 * threshold` matches, for every two-decimal threshold.

    The stubbed scorer returns the exact product as a float. The drifted
    `100 * threshold` (8 of the 99 two-decimal thresholds round away from
    it, 5 of them upward) must not reject a pair sitting exactly on the
    boundary.
    """
    exact_score = float(Decimal(str(threshold)) * 100)
    deduper = lk.fuzzy(threshold=threshold)
    monkeypatch.setattr(deduper, "get_scorer", lambda: lambda s1, s2, **kwargs: exact_score)

    pairs = list(deduper._gen_similarity_pairs(pa.array(["x", "y"])))

    assert pairs == [(0, 1)]


def test_fuzzy_does_not_match_below_the_threshold(monkeypatch):
    """A score more than the tolerance below the boundary never matches."""
    deduper = lk.fuzzy(threshold=0.95)
    monkeypatch.setattr(deduper, "get_scorer", lambda: lambda s1, s2, **kwargs: 94.0)

    assert list(deduper._gen_similarity_pairs(pa.array(["x", "y"]))) == []


def test_fuzzy_matches_a_real_score_exactly_at_the_threshold():
    """`fuzz.ratio("abcd", "abce")` is exactly 75.0, so threshold 0.75 matches and 0.76 does not."""
    assert fuzz.ratio("abcd", "abce") == 75.0

    pairs = list(lk.fuzzy(threshold=0.75)._gen_similarity_pairs(pa.array(["abcd", "abce"])))

    assert pairs == [(0, 1)]

    pairs = list(lk.fuzzy(threshold=0.76)._gen_similarity_pairs(pa.array(["abcd", "abce"])))

    assert pairs == []


def test_cosine_matches_a_pair_exactly_at_the_threshold():
    """Rows [1, 1] and [1, 0] have similarity `1/sqrt(2)`; a threshold equal to it must match."""
    threshold = float(1.0 / np.sqrt(2.0))

    pairs = list(lk.cosine(threshold=threshold)._gen_similarity_pairs(pa.table({"x": [1.0, 1.0], "y": [1.0, 0.0]})))

    assert pairs == [(0, 1)]


def test_cosine_does_not_match_below_the_threshold():
    threshold = float(math.nextafter(1.0 / np.sqrt(2.0), math.inf))

    pairs = list(lk.cosine(threshold=threshold)._gen_similarity_pairs(pa.table({"x": [1.0, 1.0], "y": [1.0, 0.0]})))

    assert pairs == []


def test_jaccard_matches_a_pair_exactly_at_the_threshold():
    """Two rows sharing 2 of their 4 members have jaccard similarity exactly 0.5."""
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": ["c", "d"]})

    pairs = list(lk.jaccard(threshold=0.5)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_jaccard_does_not_match_below_the_threshold():
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": ["c", "d"]})

    pairs = list(lk.jaccard(threshold=math.nextafter(0.5, math.inf))._gen_similarity_pairs(table))

    assert pairs == []


def test_tfidf_matches_a_pair_exactly_at_the_threshold():
    """A pair whose similarity equals the threshold matches.

    The threshold is lowered by one ULP before it reaches the delegate,
    whose comparison is strict, so a pair stored at exactly the threshold
    survives.
    """
    values = ["a b", "a c", "a b c"]

    probe = lk.tfidf(ngram=1, topn=5, threshold=math.nextafter(0.0, 1.0))
    sparse = probe._get_sparse_matrix(values).tocoo()
    row, col, value = max(
        ((int(r), int(c), float(v)) for r, c, v in zip(sparse.row, sparse.col, sparse.data, strict=False) if r < c),
        key=lambda item: item[2],
    )

    pairs = list(lk.tfidf(ngram=1, topn=5, threshold=value)._gen_similarity_pairs(pa.array(values)))

    assert (row, col) in pairs

    above = math.nextafter(value, math.inf)
    pairs = list(lk.tfidf(ngram=1, topn=5, threshold=above)._gen_similarity_pairs(pa.array(values)))

    assert (row, col) not in pairs


###################
#  token_overlap  #
###################


def test_token_overlap_containment_matches_where_jaccard_would_not():
    """The spec's headline case: a 2-member set fully contained in a 4-member set.

    Containment is 2/2 = 1.0, so it matches at threshold 0.95 where the
    jaccard ratio would be 2/4 = 0.5.
    """
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": [None, "c"], "w": [None, "d"]})

    assert list(lk.token_overlap(threshold=0.95)._gen_similarity_pairs(table)) == [(0, 1)]
    assert list(lk.jaccard(threshold=0.95)._gen_similarity_pairs(table)) == []


def test_token_overlap_subset_in_the_later_row_also_matches():
    """The denominator is min(|A|,|B|) regardless of which row holds the subset.

    The headline case with the rows swapped: a regression using
    len(sets[idx]) as the denominator would score this pair 2/4 and
    reject it at 0.95.
    """
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": ["c", None], "w": ["d", None]})

    assert list(lk.token_overlap(threshold=0.95)._gen_similarity_pairs(table)) == [(0, 1)]


def test_token_overlap_matches_a_pair_exactly_at_the_threshold():
    """A pair sharing 1 of 2 members of the smaller set has containment exactly 0.5."""
    table = pa.table({"x": ["a", "a"], "y": ["b", "c"], "z": [None, "d"], "w": [None, "e"]})

    pairs = list(lk.token_overlap(threshold=0.5)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_token_overlap_does_not_match_below_the_threshold():
    table = pa.table({"x": ["a", "a"], "y": ["b", "c"], "z": [None, "d"], "w": [None, "e"]})

    pairs = list(lk.token_overlap(threshold=math.nextafter(0.5, math.inf))._gen_similarity_pairs(table))

    assert pairs == []


def test_token_overlap_matches_in_either_direction():
    """Either direction clearing the threshold emits the pair, once, in index order.

    Row 0's set {a} is fully contained in row 1's {a, b, c}: direction
    0->1 scores 1/1 = 1.0 even though direction 1->0 scores 1/3.
    """
    table = pa.table({"x": ["a", "a"], "y": [None, "b"], "z": [None, "c"]})

    pairs = list(lk.token_overlap(threshold=0.95)._gen_similarity_pairs(table))

    assert pairs == [(0, 1)]


def test_token_overlap_two_all_null_rows_do_not_match():
    """The empty-intersection short-circuit: two rows with no members match nothing."""
    table = pa.table({"x": [None, None], "y": [None, None]})

    assert list(lk.token_overlap(threshold=0.01)._gen_similarity_pairs(table)) == []


def test_token_overlap_one_all_null_row_does_not_match_a_populated_row():
    """An all-null row's set is empty, so its intersection with anything is empty."""
    table = pa.table({"x": [None, "a"], "y": [None, "b"]})

    assert list(lk.token_overlap(threshold=0.01)._gen_similarity_pairs(table)) == []


def test_token_overlap_excludes_none_members_from_both_sets():
    """A None member neither matches a value nor counts toward min(|A|, |B|).

    Dropping the None members from both rows leaves {a, b} against
    {a, b, c}: containment 2/2 = 1.0, so the pair matches at 0.95.
    """
    table = pa.table({"x": ["a", "a"], "y": ["b", "b"], "z": [None, "c"]})

    assert list(lk.token_overlap(threshold=0.95)._gen_similarity_pairs(table)) == [(0, 1)]


def test_token_overlap_is_not_tokenized():
    """Row sets are built from cell values as they are, like jaccard: "a b" and "ab" differ."""
    table = pa.table({"x": ["a b", "ab"]})

    assert list(lk.token_overlap(threshold=0.01)._gen_similarity_pairs(table)) == []


def test_token_overlap_groups_transitively_through_union_find(mock_df):
    """A->B and B->C each clear the threshold, so all three merge into one group.

    Row sets: {a} vs {a, b} is containment 1.0; {a, b} vs {a, b, c} is
    containment 1.0; {a} vs {a, b, c} is also 1.0.
    """
    mock_df.get_array = Mock(
        return_value=pa.table(
            {
                "x": ["a", "a", "a"],
                "y": [None, "b", "b"],
                "z": [None, None, "c"],
            }
        )
    )

    uf, n = lk.token_overlap(threshold=0.95).set_frame(mock_df).build_union_find(("x", "y", "z"), [])

    assert n == 3
    assert uf[0] == uf[1] == uf[2]


def test_token_overlap_requires_compound_tuple_columns():
    with pytest.raises(InvalidArgumentTypeError):
        lk.token_overlap().validate("name")

    with pytest.raises(InvalidArgumentTypeError):
        lk.token_overlap().validate(("name", 3))


@pytest.mark.parametrize("threshold", ["0.5", None, 0.0, 1.0, -0.5])
def test_token_overlap_rejects_thresholds_outside_the_exclusive_range(threshold):
    with pytest.raises((InvalidArgumentTypeError, InvalidArgumentValueError)):
        lk.token_overlap(threshold=threshold)


def test_token_overlap_str_renders_threshold():
    assert "token_overlap(" in str(lk.token_overlap(threshold=0.75))
    assert "threshold=0.75" in str(lk.token_overlap(threshold=0.75))
