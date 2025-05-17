from __future__ import annotations
from unittest.mock import Mock, patch, call

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from dupegrouper.base import _wrap
from dupegrouper.definitions import TMP_ATTR, GROUP_ID
from dupegrouper.strategies.tfidf import TfIdf


###################
# test _vectorize #
###################


def test_vectorize_with_int():
    tfidf = TfIdf()
    vec = tfidf._vectorize(3)
    assert isinstance(vec, TfidfVectorizer)
    assert vec.ngram_range == (3, 3)


def test_vectorize_with_tuple():
    tfidf = TfIdf()
    vec = tfidf._vectorize((2, 5))
    assert isinstance(vec, TfidfVectorizer)
    assert vec.ngram_range == (2, 5)


def test_vectorize_invalid_type():
    tfidf = TfIdf()

    with pytest.raises(TypeError, match="ngram must be of type int or a length 2 tuple of integers"):
        tfidf._vectorize("invalid input e.g. this string")


####################
# DEDUPE UNIT TEST #
####################


def test_dedupe_unit():
    attr = "address"
    dummy_array = np.array(["foo", "bar", "bar"])

    tfidf = TfIdf(ngram=(1, 1), tolerance=0.2, topn=2)

    # mock for wrapped_df
    mock_wrapped_df = Mock()
    mock_wrapped_df.get_col.return_value = dummy_array
    tfidf.wrapped_df = mock_wrapped_df

    with patch.object(tfidf, "_vectorize", return_value="dummy-vectorizer") as mock_vec, patch.object(
        tfidf, "_get_similarities_matrix", return_value="dummy-matrix"
    ) as mock_sim_matrix, patch.object(
        tfidf, "_get_matches_array", return_value=(np.array([0]), np.array([1]), np.array([0.95]))
    ) as mock_matches_array, patch.object(
        tfidf, "_gen_map", return_value=iter([{"bar": "bar"}])
    ) as mock_gen_map, patch.object(
        tfidf, "assign_group_id", return_value=mock_wrapped_df
    ) as mock_assign_group_id:

        mock_wrapped_df.map_dict.return_value = [None, "bar", "bar"]
        mock_wrapped_df.fill_na.return_value = ["foo", "bar", "bar"]
        mock_wrapped_df.put_col.return_value = mock_wrapped_df
        mock_wrapped_df.assign_group_id.return_value = mock_wrapped_df
        mock_wrapped_df.drop_col.return_value = mock_wrapped_df

        result = tfidf.dedupe(attr)

        # Assertions
        mock_vec.assert_called_once_with((1, 1))

        args, _ = mock_sim_matrix.call_args
        assert args[0] == "dummy-vectorizer"
        np.testing.assert_array_equal(args[1], dummy_array)

        args, _ = mock_matches_array.call_args
        assert args[0] == "dummy-matrix"
        np.testing.assert_array_equal(args[1], dummy_array)

        mock_gen_map.assert_called_once()

        mock_wrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})
        mock_wrapped_df.fill_na.assert_called_once()

        # second put call is part of assign_group_id which in another unit test
        put_col_call = mock_wrapped_df.put_col.call_args_list[0]
        assert put_col_call == call(TMP_ATTR, ["foo", "bar", "bar"])

        mock_assign_group_id.assert_called_once()
        mock_wrapped_df.drop_col.assert_called_once()

        assert result == mock_wrapped_df


##################################
# DEDUPE NARROW INTEGRATION TEST #
##################################


tfidf_parametrize_data = [
    # i.e. no deduping, by definition
    ({"ngram": (1, 1), "tolerance": 0, "topn": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # progressive tolerance
    ({"ngram": (1, 1), "tolerance": 0.05, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"ngram": (1, 1), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13]),
    ({"ngram": (1, 1), "tolerance": 0.35, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 6]),
    ({"ngram": (1, 1), "tolerance": 0.50, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 11, 6]),
    ({"ngram": (1, 1), "tolerance": 0.65, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 11, 6]),
    ({"ngram": (1, 1), "tolerance": 0.85, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 11, 6]),
    # progressive ngram @ 0.2
    ({"ngram": (1, 2), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (1, 3), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (2, 3), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (2, 2), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (3, 3), "tolerance": 0.20, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    # progressive ngram @ 0.4
    ({"ngram": (1, 2), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 3, 2, 2, 3, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (1, 3), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (2, 3), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (2, 2), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (3, 3), "tolerance": 0.40, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
]


@pytest.mark.parametrize("input, output", tfidf_parametrize_data)
def test_dedupe_integrated(input, output, dataframe, helpers):

    df, spark, id_col = dataframe

    if spark:
        # i.e. Spark DataFrame -> Spark list[Row]
        df = df.collect()

    tfidf = TfIdf(**input)
    tfidf.with_frame(_wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, GROUP_ID) == output
