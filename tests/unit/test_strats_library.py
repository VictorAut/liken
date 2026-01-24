from unittest.mock import Mock

import numpy as np
import pytest

from enlace import (
    cosine,
    exact,
    fuzzy,
    jaccard,
    lsh,
    tfidf,
)
from enlace._strats_library import (
    LSH,
    BaseStrategy,
    Cosine,
    Exact,
    Fuzzy,
    Jaccard,
    StrContains,
    StrEndsWith,
    StrStartsWith,
    TfIdf,
)
from enlace.rules import (
    str_contains,
    str_endswith,
    str_startswith,
)


############
# Fixtures #
###########


@pytest.fixture
def mock_df():
    """
    Minimal LocalDF. Only methods used by strategies defined.
    """
    df = Mock()
    df._get_col.return_value = np.array([1, 2, 3], dtype=object)
    df._get_cols.return_value = np.array([[1], [2], [3]], dtype=object)
    df.put_col.return_value = df
    df.get_array.return_value = np.array([1, 2, 3], dtype=object) # here as a placeholder
    return df


##############################
# BaseStrategy core behavior #
##############################


def test_set_frame_sets_wrapped_df(mock_df):
    strat = BaseStrategy()
    returned = strat.set_frame(mock_df)
    assert returned is strat
    assert strat.wdf is mock_df


def test_gen_similarity_pairs_not_implemented():
    strat = BaseStrategy()
    with pytest.raises(NotImplementedError):
        list(strat._gen_similarity_pairs(np.array([])))


################
# canonicalize #
################


def test_canonicalize_puts_canonical_id(mock_df):
    strat = BaseStrategy()
    strat.set_frame(mock_df)

    strat.wdf.get_array = Mock(
        side_effect=[
            np.array([10, 20, 30], dtype=object),
            np.array(["a", "a", "b"], dtype=object),
        ]
    )

    strat.wdf.get_canonical = Mock(side_effect=[np.array([10, 20, 30], dtype=object)])

    components = {
        0: [0, 1],
        2: [2],
    }

    result = strat.canonicalizer(components=components, drop_duplicates=False, keep="first")

    mock_df.put_col.assert_called_once()
    assert result is mock_df


####################
# ColumnArrayMixin #
####################


def test_column_array_mixin_str_column(mock_df):
    strat = Exact().set_frame(mock_df)
    arr = strat.wdf.get_array("a")
    mock_df.get_array.assert_called_once_with("a")
    assert isinstance(arr, np.ndarray)


def test_column_array_mixin_tuple_column(mock_df):
    strat = Exact().set_frame(mock_df)
    arr = strat.wdf.get_array(("a", "b"))
    mock_df.get_array.assert_called_once_with(("a", "b"))
    assert isinstance(arr, np.ndarray)


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


############################
# Public factory functions #
############################


@pytest.mark.parametrize(
    "factory, cls",
    [
        (exact, Exact),
        (lambda: str_startswith("a"), StrStartsWith),
        (lambda: str_endswith("a"), StrEndsWith),
        (lambda: str_contains("a"), StrContains),
        (fuzzy, Fuzzy),
        (tfidf, TfIdf),
        (lsh, LSH),
        (jaccard, Jaccard),
        (cosine, Cosine),
    ],
)
def test_public_factories_return_correct_type(factory, cls):
    strat = factory()
    assert isinstance(strat, cls)
