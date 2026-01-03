from __future__ import annotations
from unittest.mock import Mock, patch, call

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from dupegrouper.base import wrap
from dupegrouper.definitions import TMP_ATTR_LABEL, CANONICAL_ID
from dupegrouper.strategies import TfIdf



####################
# DEDUPE UNIT TEST #
####################


# def test_dedupe_unit():
#     attr = "address"
#     dummy_array = np.array(["foo", "bar", "bar"])

#     tfidf = TfIdf(ngram=(1, 1), threshold=0.2, topn=2)

#     # mock for wrapped_df
#     mockwrapped_df = Mock()
#     mockwrapped_df.get_col.return_value = dummy_array
#     tfidf.wrapped_df = mockwrapped_df

#     with patch.object(tfidf, "_vectorize", return_value="dummy-vectorizer") as mock_vec, patch.object(
#         tfidf, "_get_similarities_matrix", return_value="dummy-matrix"
#     ) as mock_sim_matrix, patch.object(
#         tfidf, "_get_matches_array", return_value=(np.array([0]), np.array([1]), np.array([0.95]))
#     ) as mock_matches_array, patch.object(
#         tfidf, "_gen_map", return_value=iter([{"bar": "bar"}])
#     ) as mock_gen_map, patch.object(
#         tfidf, "canonicalize", return_value=mockwrapped_df
#     ) as mock_canonicalize:

#         mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
#         mockwrapped_df.fill_na.return_value = ["foo", "bar", "bar"]
#         mockwrapped_df.put_col.return_value = mockwrapped_df
#         mockwrapped_df.canonicalize.return_value = mockwrapped_df
#         mockwrapped_df.drop_col.return_value = mockwrapped_df

#         result = tfidf.dedupe(attr)

#         # Assertions
#         mock_vec.assert_called_once_with((1, 1))

#         args, _ = mock_sim_matrix.call_args
#         assert args[0] == "dummy-vectorizer"
#         np.testing.assert_array_equal(args[1], dummy_array)

#         args, _ = mock_matches_array.call_args
#         assert args[0] == "dummy-matrix"
#         np.testing.assert_array_equal(args[1], dummy_array)

#         mock_gen_map.assert_called_once()

#         mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "bar"})
#         mockwrapped_df.fill_na.assert_called_once()

#         # second put call is part of canonicalize which in another unit test
#         put_col_call = mockwrapped_df.put_col.call_args_list[0]
#         assert put_col_call == call(TMP_ATTR_LABEL, ["foo", "bar", "bar"])

#         mock_canonicalize.assert_called_once()
#         mockwrapped_df.drop_col.assert_called_once()

#         assert result == mockwrapped_df


##################################
# DEDUPE NARROW INTEGRATION TEST #
##################################


tfidf_parametrize_data = [
    # i.e. no deduping, by definition
    ({"ngram": (1, 1), "threshold": 1, "topn": 1}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # progressive threshold
    ({"ngram": (1, 1), "threshold": 0.95, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ({"ngram": (1, 1), "threshold": 0.80, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 13]),
    ({"ngram": (1, 1), "threshold": 0.65, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 12, 2]),
    ({"ngram": (1, 1), "threshold": 0.50, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 5, 2]),
    ({"ngram": (1, 1), "threshold": 0.35, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 5, 2]),
    ({"ngram": (1, 1), "threshold": 0.15, "topn": 2}, [1, 2, 3, 3, 5, 2, 3, 2, 1, 1, 5, 5, 2]),
    # progressive ngram @ 0.2
    ({"ngram": (1, 2), "threshold": 0.80, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (1, 3), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (2, 3), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (2, 2), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    ({"ngram": (3, 3), "threshold": 0.80, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
    # progressive ngram @ 0.4
    ({"ngram": (1, 2), "threshold": 0.60, "topn": 2}, [1, 2, 3, 3, 2, 2, 3, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (1, 3), "threshold": 0.60, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (2, 3), "threshold": 0.60, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (2, 2), "threshold": 0.60, "topn": 2}, [1, 2, 3, 3, 5, 6, 7, 2, 1, 1, 11, 12, 13]),
    ({"ngram": (3, 3), "threshold": 0.60, "topn": 2}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # No deduping!
]


@pytest.mark.parametrize("input, output", tfidf_parametrize_data)
def test_dedupe_integrated(input, output, dataframe, helpers):

    df, spark, id_col = dataframe

    if spark:
        # i.e. Spark DataFrame -> Spark list[Row]
        df = df.collect()

    tfidf = TfIdf(**input)
    tfidf.with_frame(wrap(df, id_col))

    df = tfidf.dedupe("address").unwrap()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == output
