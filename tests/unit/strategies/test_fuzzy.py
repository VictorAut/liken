from __future__ import annotations
from unittest.mock import Mock, patch, call

import numpy as np
import pytest

from dupegrouper.base import wrap
from dupegrouper.definitions import TMP_ATTR_LABEL, CANONICAL_ID
from dupegrouper.strategies import Fuzzy


####################
# DEDUPE UNIT TEST #
####################


# def test_dedupe_unit():
#     attr = "address"
#     dummy_array = np.array(["foo", "bar", "bar"])

#     tfidf = Fuzzy(threshold=0.2)

#     mockwrapped_df = Mock()clear

#     mockwrapped_df.get_col.return_value = dummy_array
#     tfidf.wrapped_df = mockwrapped_df

#     with patch.object(
#         tfidf,
#         "_fuzz_ratio",
#         return_value=85.1,
#     ) as mock_fuzz, patch.object(
#         tfidf,
#         "canonicalize",
#         return_value=mockwrapped_df,
#     ) as mock_canonicalize:

#         # Also mock wrapped_df chaining methods
#         mockwrapped_df.map_dict.return_value = [None, "bar", "bar"]
#         mockwrapped_df.put_col.return_value = mockwrapped_df
#         mockwrapped_df.canonicalize.return_value = mockwrapped_df
#         mockwrapped_df.drop_col.return_value = mockwrapped_df

#         # Run dedupe
#         result = tfidf.dedupe(attr)

#         # Assertions
#         mock_fuzz.assert_called_with("foo", "foo")

#         mockwrapped_df.map_dict.assert_called_once_with(attr, {"bar": "foo", "foo": "foo"})

#         # second put call is part of canonicalize which in another unit test
#         put_col_call = mockwrapped_df.put_col.call_args_list[0]
#         assert put_col_call == call(TMP_ATTR_LABEL, [None, "bar", "bar"])

#         mock_canonicalize.assert_called_once()
#         mockwrapped_df.drop_col.assert_called_once()

#         assert result == mockwrapped_df

