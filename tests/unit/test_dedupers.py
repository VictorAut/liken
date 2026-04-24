from unittest.mock import Mock

import pyarrow as pa
import pytest

from liken import cosine
from liken import exact
from liken import fuzzy
from liken import jaccard
from liken import lsh
from liken import str_contains
from liken import str_endswith
from liken import str_startswith
from liken import tfidf
from liken.core.deduper import BaseDeduper
from liken.dedupers.cosine import Cosine
from liken.dedupers.exact import Exact
from liken.dedupers.fuzzy import Fuzzy
from liken.dedupers.jaccard import Jaccard
from liken.dedupers.lsh import LSH
from liken.dedupers.str_contains import StrContains
from liken.dedupers.str_endswith import StrEndsWith
from liken.dedupers.str_startswith import StrStartsWith
from liken.dedupers.tfidf import TfIdf


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
    deduper = factory()
    assert isinstance(deduper, cls)
