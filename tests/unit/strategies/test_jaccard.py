from __future__ import annotations
from unittest.mock import Mock, patch, call

import numpy as np
import pytest

from dupegrouper.base import _wrap
from dupegrouper.definitions import TMP_ATTR_LABEL, CANONICAL_ID
from dupegrouper.strategies.compound_columns import Jaccard


##################################
# DEDUPE NARROW INTEGRATION TEST #
##################################


fuzzy_parametrize_data = [
    # i.e. no deduping
    ({"tolerance": 0}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    # progressive deduping
    ({"tolerance": 0.35}, [1, 1, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 13]),
    ({"tolerance": 0.65}, [1, 1, 3, 4, 5, 5, 1, 4, 9, 10, 1, 1, 5]),
    ({"tolerance": 0.85}, [1, 1, 3, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1]),
    ({"tolerance": 0.95}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
]


@pytest.mark.parametrize("input, output", fuzzy_parametrize_data)
def test_dedupe_integrated(input, output, dataframe, helpers):

    df, spark, id_col = dataframe

    if spark:
        # i.e. Spark DataFrame -> Spark list[Row]
        df = df.collect()

    tfidf = Jaccard(**input)
    tfidf.with_frame(_wrap(df, id_col))

    compound_col = (
        "account",
        "birth_country",
        "martial_status",
        "number_children",
        "property_type",
    )

    df = tfidf.dedupe(compound_col).unwrap()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == output
