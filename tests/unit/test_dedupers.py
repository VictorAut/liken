import math
from datetime import date
from decimal import Decimal
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
    [-1, 1.0, "1", True, None],
    ids=["negative", "float", "str", "bool", "none"],
)
def test_date_range_rejects_bad_window(window):
    with pytest.raises((InvalidArgumentTypeError, InvalidArgumentValueError)):
        lk.date_range(format="%Y-%m-%d", window=window)


@pytest.mark.parametrize(
    "window",
    [0, 1, 30],
    ids=["zero", "one", "thirty"],
)
def test_date_range_accepts_valid_window(window):
    assert lk.date_range(format="%Y-%m-%d", window=window)._window == window


def test_date_range_default_window_is_zero():
    assert lk.date_range(format="%Y-%m-%d")._window == 0


@pytest.mark.parametrize(
    "values, window, expected",
    [
        # same date: star-shaped from the first index (predicate semantics)
        (["2024-01-05", "2024-01-05", "2024-01-05"], 0, [(0, 1), (0, 2)]),
        # one day apart at window=1
        (["2024-01-05", "2024-01-06"], 1, [(0, 1)]),
        # two days apart at window=1: rejected
        (["2024-01-05", "2024-01-07"], 1, []),
        # inclusive boundary: exactly window days apart matches; 1 day
        # apart is inside window=2
        (["2024-01-05", "2024-01-06", "2024-01-07"], 2, [(0, 1), (0, 2), (1, 2)]),
        # window=0 is same-date equality: adjacent dates never pair
        (["2024-01-05", "2024-01-06"], 0, []),
        # duplicates at window=1 all pair
        (["2024-03-01", "2024-03-01", "2024-03-02"], 1, [(0, 1), (0, 2), (1, 2)]),
        # far apart at any window: nothing matches
        (["2024-01-01", "2024-06-01"], 30, []),
    ],
)
def test_date_range_matches_pairs_within_window(values, window, expected):
    deduper = lk.date_range(format="%Y-%m-%d", window=window)
    deduper.validate("date")

    pairs = sorted(deduper._gen_similarity_pairs(pa.array(values)))

    assert pairs == expected


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
    with pytest.raises(InvalidArgumentTypeError, match="string"):
        deduper = lk.date_range(format="%Y-%m-%d")
        deduper.validate("n")
        list(deduper._gen_similarity_pairs(pa.array([1, 2])))


def test_date_range_boolean_array_rejected_at_execution():
    with pytest.raises(InvalidArgumentTypeError, match="string"):
        deduper = lk.date_range(format="%Y-%m-%d")
        deduper.validate("flag")
        list(deduper._gen_similarity_pairs(pa.array([True, False])))


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
