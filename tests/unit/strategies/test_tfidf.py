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
