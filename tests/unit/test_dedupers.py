from unittest.mock import Mock

import pyarrow as pa
import pytest

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
    df.get_array.return_value = pa.array([1, 2, 3])  # here as a placeholder
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


def test_edit_distance_null_placeholder_matches_only_null_placeholder():
    """A coalesced null ("na") must not match real values near it, matching
    fuzzy's behaviour where nulls do not match anything but each other."""
    deduper = EditDistance(max_distance=2)

    # "na" stands for a coalesced null; the others are real values
    pairs = list(deduper._gen_similarity_pairs(pa.array(["na", "na", "NA", "", "nna"])))

    # the nulls match each other; the real values "NA" and "" are 2 edits
    # apart and match each other; no placeholder value pairs with a real one
    assert pairs == [(0, 1), (2, 3)]
