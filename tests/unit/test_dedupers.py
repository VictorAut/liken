import math
from decimal import Decimal
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest
from rapidfuzz import fuzz

import liken as lk
from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.dedupers.edit_distance import EditDistance
from liken.dedupers.exact import Exact
from liken.dedupers.fuzzy import Fuzzy
from liken.dedupers.isin import isin
from liken.dedupers.jaccard import Jaccard
from liken.dedupers.str_contains import StrContains
from liken.dedupers.str_startswith import StrStartsWith


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

    result = deduper.canonicalizer(components=components, drop_duplicates=False, keep="first")

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
    with pytest.raises(ValueError):
        StrStartsWith("x").validate(("a", "b"))


def test_compound_column_validation_accepts_tuple():
    Jaccard().validate(("a", "b"))


def test_compound_column_validation_rejects_str():
    with pytest.raises(ValueError):
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
    ("isin", lk.isin("london"), ["isin(", "values='london'"]),
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
    [(0.0, [(0, 2), (1, 3)]), (0.55, [(1, 3)])],
    ids=["threshold-zero", "just-above-the-value-pair"],
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


def test_fuzzy_unknown_scorer_raises_key_error():
    """An unregistered scorer name surfaces a KeyError, not a silent fallback."""
    with pytest.raises(KeyError):
        Fuzzy(threshold=0.95, scorer="no_such_scorer").get_scorer()


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
    with pytest.raises(ValueError):
        lk.edit_distance(max_distance=-1)


def test_edit_distance_non_integer_max_distance_raises_value_error():
    with pytest.raises(ValueError):
        lk.edit_distance(max_distance=1.5)


def test_edit_distance_requires_single_string_column():
    with pytest.raises(ValueError):
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
        {"min": float("nan")},
        {"max": float("nan")},
        {"min": float("nan"), "max": 5},
    ],
    ids=["min-bool", "max-bool", "min-false", "min-str", "max-str", "min-nan", "max-nan", "min-nan-with-max"],
)
def test_num_range_rejects_non_real_bounds(kwargs):
    """A bound that is not a real number — bool, str, NaN — is refused at construction."""
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
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

    probe = lk.tfidf(ngram=1, topn=5, threshold=0.0)
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
