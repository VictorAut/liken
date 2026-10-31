"""Liken deduper unit tests"""

from unittest.mock import Mock

import pyarrow as pa
import pytest
from typing_extensions import override

import liken as lk
from liken.constants import CANONICAL_ID
from liken.core.deduper import BaseDeduper
from liken.core.deduper import PredicateDeduper
from liken.dedupers.exact import Exact
from liken.dedupers.isin import isin
from liken.dedupers.jaccard import Jaccard
from liken.dedupers.str_contains import StrContains
from liken.dedupers.str_startswith import StrStartsWith
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError


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


#####################################################
# scalar path honours `case` (WI072)                #
#####################################################


class _FallbackOnlyPredicate:
    """Wraps a predicate deduper subclass whose `_vectorized_matches`
    returns `None`, forcing the scalar path while keeping the same
    `_matches`."""

    @staticmethod
    def fallback_of(deduper):
        class _Fallback(type(deduper)):
            @override
            def _vectorized_matches(self, array: pa.Array) -> pa.Array | None:
                return None

        return _Fallback(**deduper._init_kwargs)


def _match_decisions(deduper, values):
    """Match decision per value as a bool.

    The vectorised mask is null at a missing value (the negated-path
    contract); the scalar path answers `False` there. Null means "not a
    match", so both sides normalise to bool for comparison.
    """
    array = pa.array(values, type=pa.string())
    mask = deduper._vectorized_matches(array)
    if mask is None:
        return [deduper._matches(v) for v in values]
    return [bool(mask[i].as_py()) for i in range(len(values))]


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (lk.str_startswith, {"pattern": "x"}),
        (lk.str_endswith, {"pattern": "x"}),
        (lk.str_contains, {"pattern": "x"}),
        (lk.str_contains, {"pattern": "X", "regex": True}),
    ],
    ids=["startswith", "endswith", "contains", "contains-regex"],
)
@pytest.mark.parametrize("case", [True, False], ids=["case-sensitive", "case-insensitive"])
def test_scalar_matches_agrees_with_vectorized_matches(factory, kwargs, case):
    """The scalar path must honour `case` exactly as the vectorised path
    does, on any value. WI072: before the fix, `case=False` scalar
    returned `False` for mixed-case values the vectorised path matched.
    """
    values = ["Xab", "xaB", "abc", "zzz", None]

    vectorized = factory(case=case, **kwargs)
    fallback = _FallbackOnlyPredicate.fallback_of(vectorized)

    assert _match_decisions(fallback, values) == _match_decisions(vectorized, values)


def test_scalar_matches_missing_is_false_every_case():
    """The negated-deduper contract: a missing value never satisfies a
    predicate, on any case setting."""
    for factory, kwargs in [
        (lk.str_startswith, {"pattern": "x"}),
        (lk.str_endswith, {"pattern": "x"}),
        (lk.str_contains, {"pattern": "x"}),
    ]:
        for case in [True, False]:
            assert factory(case=case, **kwargs)._matches(None) is False


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
