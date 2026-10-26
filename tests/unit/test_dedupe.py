from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

import liken as lk
from liken.constants import CANONICAL_ID
from liken.constants import INTERNAL_CANONICAL_ID
from liken.constants import INTERNAL_CANONICAL_ID_COLLISION_MSG
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError
from liken.liken import Dedupe
from liken.validations import validate_keep_arg


@patch("liken.liken.CollectionsManager")
def test_no_apply_still_has_exact_apply(
    mock_sm,
    dataframe,
):

    sm = mock_sm.return_value
    sm.has_applies = False
    sm.is_sequential_applied = True
    sm.get.return_value = {}
    sm.reset.return_value = None

    lk = Dedupe(dataframe)
    lk.canonicalize("address")  # no apply!

    sm.apply.assert_called_once()


@patch("liken.liken.CollectionsManager")
def test_no_apply_still_has_exact_apply_on_drop_duplicates(
    mock_sm,
    dataframe,
):

    sm = mock_sm.return_value
    sm.has_applies = False
    sm.is_sequential_applied = True
    sm.get.return_value = {}
    sm.reset.return_value = None

    lk = Dedupe(dataframe)
    lk.drop_duplicates("address")  # no apply!

    sm.apply.assert_called_once()


# canonicals


def test_re_execution_after_drop_duplicates_reapplies_exact():
    """exact fallback fires on every execute, not only the first.

    The second execute runs on a clean frame: `drop_duplicates` reset the
    collection, so `canonicalize` must re-apply the exact fallback and group
    the two equal "a" values.
    """
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": ["p", "q", "r"]})

    dedupe = Dedupe(df)
    dedupe.apply(lk.exact()).drop_duplicates("b")  # no duplicates in "b"

    dedupe.canonicalize("a")

    assert list(dedupe.collect()["canonical_id"]) == [0, 0, 2]


def test_canonicals_rejects_n_below_two(dataframe):
    dedupe = Dedupe(dataframe)
    with pytest.raises(InvalidArgumentValueError, match=r"`n` must be an integer in the range \[2, inf\), got 1"):
        dedupe.canonicals(n=1)


@pytest.mark.parametrize("n", ["2", 2.5, True, None])
def test_canonicals_rejects_non_integer_n(n, dataframe):
    """`n` is type-checked before the bound."""
    dedupe = Dedupe(dataframe)
    with pytest.raises(InvalidArgumentTypeError, match="`n` must be an integer, got"):
        dedupe.canonicals(n=n)


def test_canonicalize_rejects_empty_id():
    """an empty `source_id` previously fell through to autoincrement."""
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(InvalidArgumentValueError, match="`source_id` must be a non-empty string, got ''"):
        dedupe.canonicalize("a", source_id="")


def test_canonicalize_rejects_non_str_id():
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(InvalidArgumentTypeError, match="`source_id` must be of type 'str', got 'int'"):
        dedupe.canonicalize("a", source_id=1)


def test_canonicalize_rejects_non_bool_drop_duplicates():
    """`drop_duplicates=0` was accepted silently and dropped nothing."""
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(InvalidArgumentTypeError, match="`drop_duplicates` must be of type 'bool', got 'int'"):
        dedupe.canonicalize("a", drop_duplicates=0)


def test_synthesize_before_canonicalize_raises():
    """`synthesize` requires a prior `canonicalize`; the state error
    comes before any frame work."""
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(RuntimeError, match=r"Run `\.canonicalize\(\)` first"):
        dedupe.synthesize()


def test_canonicals_before_canonicalize_raises(dataframe):
    dedupe = Dedupe(dataframe)
    with pytest.raises(RuntimeError, match="No canonical_id counts found"):
        dedupe.canonicals()


# dispatch


def test_list_of_rows_rejected_at_entrypoint():
    pytest.importorskip("pyspark")
    from pyspark.sql import Row

    with pytest.raises(InvalidArgumentValueError, match=r"pyspark\.sql\.DataFrame"):
        Dedupe([Row(id=1), Row(id=2)])


def test_unsupported_dataframe_type_raises():
    with pytest.raises(InvalidArgumentValueError, match="Unsupported dataframe type: <class 'str'>"):
        Dedupe("not-a-dataframe")


# validations


@pytest.mark.parametrize("keep", ["first", "last"])
def test_validate_keep_arg_valid(keep):
    assert validate_keep_arg(keep) == keep


def test_validate_keep_arg_invalid():
    with pytest.raises(ValueError, match=r"`keep` must be one of \('first', 'last'\), got 'middle'"):
        validate_keep_arg("middle")


# Misuse of public API:


@patch("liken.liken.wrap")
def test_validate_columns_args_not_used(mock_wrap, deduplication_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="columns cannot be None"):
        lk = Dedupe(pd.DataFrame({"address": [1], "email": [2]}))
        lk.apply(deduplication_mock)
        lk.canonicalize()  # <-- shouldn't be empty


@patch("liken.liken.wrap")
def test_validate_columns_args_not_none(mock_wrap, deduplication_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="columns must be None when using the dict API"):
        lk = Dedupe(pd.DataFrame({"address": [1], "email": [2]}))
        lk.apply({"address": deduplication_mock})  # <-- label here
        lk.canonicalize("address")  # <-- so should not be used here


@patch("liken.liken.wrap")
def test_validate_columns_args_repeated(mock_wrap, deduplication_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="columns labels cannot be repeated"):
        lk = Dedupe(pd.DataFrame({"address": [1], "email": [2]}))
        lk.apply(deduplication_mock)
        lk.canonicalize(("email", "email"))  # <-- shouldn't be repeated


@patch("liken.liken.wrap")
def test_validate_columns_args_type_error_precedes_mode(mock_wrap, deduplication_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(TypeError, match="`columns` must be of type 'str' or 'tuple', got 'int'"):
        lk = Dedupe(pd.DataFrame({"address": [1], "email": [2]}))
        lk.apply(deduplication_mock)
        lk.canonicalize(123)


@patch("liken.liken.wrap")
def test_validate_columns_args_type_error_precedes_mode_in_dict(mock_wrap, deduplication_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(TypeError, match="`columns` must be of type 'str' or 'tuple', got 'list'"):
        lk = Dedupe(pd.DataFrame({"address": [1], "email": [2]}))
        lk.apply({"address": deduplication_mock})
        lk.canonicalize(["address"])


@patch("liken.liken.wrap")
def test_validate_columns_args_non_str_tuple_member_type_error(mock_wrap, deduplication_mock):
    """labels must be str; a non-str label is a type error, not accepted."""
    mock_wrap.return_value = Mock()

    with pytest.raises(TypeError, match="`columns` members must be of type 'str', got 'int'"):
        lk = Dedupe(pd.DataFrame({"address": [1], "email": [2]}))
        lk.apply(deduplication_mock)
        lk.canonicalize((0,))


# Boundary checks (U13, U46, U51):


def test_execute_rejects_nonexistent_column_label():
    """`columns` labels are existence-checked at the frame boundary."""
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(InvalidArgumentValueError, match="column 'nope' not found in the dataframe"):
        dedupe.drop_duplicates("nope")


def test_execute_rejects_nonexistent_tuple_member():
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(InvalidArgumentValueError, match="column 'nope' not found in the dataframe"):
        dedupe.drop_duplicates(("a", "nope"))


def test_canonicalize_rejects_nonexistent_id():
    """`source_id` names an existing column."""
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(InvalidArgumentValueError, match="column 'nope' not found in the dataframe"):
        dedupe.canonicalize("a", source_id="nope")


def test_apply_rejects_nonexistent_dict_tuple_member():
    """dict tuple-key members are existence-checked at apply."""
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(InvalidArgumentValueError, match="column 'nope' not found in the dataframe"):
        dedupe.apply({("a", "nope"): lk.exact()})


def test_execute_rejects_zero_column_frame():
    """a zero-column frame is rejected at the execute boundary."""
    dedupe = Dedupe(pd.DataFrame())

    with pytest.raises(ValueError, match="dataframe has no columns"):
        dedupe.drop_duplicates()


def test_execute_accepts_zero_row_frame():
    """a zero-row frame stays accepted."""
    dedupe = Dedupe(pd.DataFrame({"a": [], "b": []}))

    dedupe.drop_duplicates("a")  # must not raise


def test_execute_rejects_duplicate_frame_labels():
    """duplicate labels in the user frame are rejected at the execute
    boundary, naming the labels."""
    dedupe = Dedupe(pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"]))

    with pytest.raises(ValueError, match="dataframe has duplicate column labels: \\['a'\\]"):
        dedupe.drop_duplicates("a")


# CollectionsManager


@patch("liken.liken.CollectionsManager")
def test_apply_delegates_to_collections_manager(mock, dataframe, deduplication_mock):
    mock_sm = mock.return_value
    dedupe = Dedupe(dataframe)
    dedupe.apply({"address": deduplication_mock})
    mock_sm.apply.assert_called_once_with({"address": deduplication_mock})


# canonicalize / drop_duplicates


@patch("liken.liken.LocalExecutor")
@patch("liken.liken.wrap")
@patch("liken.liken.CollectionsManager")
def test_canonicalize_calls(
    mock_sm,
    mock_wrap,
    mock_local,
    dataframe,
    deduplication_mock,
):
    mock_wrap.return_value = Mock()
    mock_executor = mock_local.return_value
    mock_sm = mock_sm.return_value
    mock_sm.get.return_value = {"address": deduplication_mock}
    mock_sm.validate_api_mode.side_effect = lambda columns: columns

    dedupe = Dedupe(dataframe)
    dedupe._executor = mock_executor
    dedupe._sm = mock_sm

    dedupe.canonicalize("address")

    mock_sm.get.assert_called_once()
    mock_executor.execute.assert_called_once_with(
        mock_wrap.return_value,
        columns="address",
        dedupers={"address": deduplication_mock},
        keep="first",
        drop_duplicates=False,
        drop_canonical_id=False,
        source_id=None,
        canonical_id=CANONICAL_ID,
    )
    mock_sm.reset.assert_called_once()


@patch("liken.liken.LocalExecutor")
@patch("liken.liken.wrap")
@patch("liken.liken.CollectionsManager")
def test_drop_duplicate_calls(
    mock_sm,
    mock_wrap,
    mock_local,
    dataframe,
    deduplication_mock,
):
    mock_wrap.return_value = Mock()
    mock_executor = mock_local.return_value
    mock_sm = mock_sm.return_value
    mock_sm.get.return_value = {"address": deduplication_mock}
    mock_sm.validate_api_mode.side_effect = lambda columns: columns

    dedupe = Dedupe(dataframe)
    dedupe._executor = mock_executor
    dedupe._sm = mock_sm

    dedupe.drop_duplicates("address")

    mock_sm.get.assert_called_once()
    mock_executor.execute.assert_called_once_with(
        mock_wrap.return_value,
        columns="address",
        dedupers={"address": deduplication_mock},
        keep="first",
        drop_duplicates=True,
        drop_canonical_id=True,
        source_id=None,
        canonical_id=INTERNAL_CANONICAL_ID,
    )
    mock_sm.reset.assert_called_once()


# Property attributes


@patch("liken.liken.CollectionsManager")
def test_dedupers_property_returns_manager_output(mock_sm, dataframe):
    mock_sm = mock_sm.return_value
    mock_sm.pretty_get.return_value = ("deduper1",)

    dedupe = Dedupe(dataframe)
    dedupe._sm = mock_sm
    assert dedupe.explain() == ("deduper1",)


# explain() with sequential dedupers


def test_explain_single_sequential_deduper_returns_that_deduper_str(dataframe):
    exact = lk.exact()
    dedupe = Dedupe(dataframe).apply(exact)

    assert dedupe.explain() == str(exact)


def test_explain_two_sequential_dedupers_returns_both_names_comma_separated(dataframe):
    exact = lk.exact()
    fuzzy = lk.fuzzy()
    dedupe = Dedupe(dataframe).apply(exact).apply(fuzzy)

    out = dedupe.explain()
    assert isinstance(out, str)
    lowered = out.lower()
    assert "exact" in lowered
    assert "fuzzy" in lowered
    assert "," in out


def test_canonicalize_rejects_canonical_id_collision():
    """`canonical_id` naming an existing column is rejected before execution."""
    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    with pytest.raises(ValueError, match="the dataframe already has a column with this label"):
        dedupe.canonicalize("a", canonical_id="b")


def test_drop_duplicates_signature_unchanged():
    """`drop_duplicates` exposes no new parameter."""
    import inspect

    params = inspect.signature(Dedupe.drop_duplicates).parameters
    assert list(params) == ["self", "columns", "keep"]


def test_internal_canonical_id_is_the_internal_label():
    assert INTERNAL_CANONICAL_ID == "__liken_canonical_private_id__"


def test_internal_canonical_id_not_exported():
    assert "INTERNAL_CANONICAL_ID" not in lk.__all__


def test_drop_duplicates_keeps_data_column_named_canonical_id():
    """A user data column named like the canonical label is never consumed.

    The drop path uses the internal label, so the user's `canonical_id`
    column survives as data.
    """
    df = pd.DataFrame({"canonical_id": [101, 102, 103], "email": ["a@x", "a@x", "b@x"]})

    out = Dedupe(df).drop_duplicates("email")

    assert list(out.columns) == ["canonical_id", "email"]
    assert list(out["canonical_id"]) == [101, 103]
    assert list(out["email"]) == ["a@x", "b@x"]


def test_drop_duplicates_keeps_custom_labelled_canonical_column():
    """Path convergence: the stored label is irrelevant to the drop path."""
    df = pd.DataFrame({"master_id": [7, 7, 8], "email": ["a@x", "a@x", "b@x"]})

    dedupe = Dedupe(df)
    dedupe._canonical_id = "master_id"

    out = dedupe.drop_duplicates("email")

    assert list(out.columns) == ["master_id", "email"]
    assert list(out["master_id"]) == [7, 8]
    assert list(out["email"]) == ["a@x", "b@x"]


def test_drop_duplicates_rejects_internal_label_collision():
    """A frame pre-carrying the internal label is rejected before execution."""
    df = pd.DataFrame({INTERNAL_CANONICAL_ID: [1, 2], "email": ["a@x", "b@x"]})

    dedupe = Dedupe(df)

    with pytest.raises(ValueError, match=INTERNAL_CANONICAL_ID_COLLISION_MSG):
        dedupe.drop_duplicates("email")


def test_canonicalize_rejects_reserved_label_argument():
    """`canonical_id=INTERNAL_CANONICAL_ID` is rejected before execution
    (Amendment 2)."""
    from liken.constants import CANONICAL_ID_RESERVED_MSG

    dedupe = Dedupe(pd.DataFrame({"a": [1, 2], "email": ["x", "y"]}))

    with pytest.raises(ValueError, match=CANONICAL_ID_RESERVED_MSG):
        dedupe.canonicalize("a", canonical_id=INTERNAL_CANONICAL_ID)


def test_canonicalize_rejects_frame_carrying_internal_label():
    """A frame pre-carrying the internal label is rejected before execution
    (Amendment 2)."""
    from liken.constants import INTERNAL_CANONICAL_ID_COLLISION_MSG

    dedupe = Dedupe(pd.DataFrame({INTERNAL_CANONICAL_ID: [1, 2], "email": ["x", "y"]}))

    with pytest.raises(ValueError, match=INTERNAL_CANONICAL_ID_COLLISION_MSG):
        dedupe.canonicalize("email")
