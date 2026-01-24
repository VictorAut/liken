"""Test that canonicalisation is only operated upon per partition"""

from __future__ import annotations

import pytest

from dupegrouper import exact
from dupegrouper._constants import CANONICAL_ID
from dupegrouper.dedupe import Dedupe


@pytest.mark.parametrize(
    "num_partitions, expected_ids",
    [
        (1, [0, 1, 2, 3, 4, 4, 6, 0, 4, 9]),
        (2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
    ids=["1 partitions", "2 partitions"],
)
def test_matrix_spark(num_partitions, expected_ids, df_spark, spark, blocking_key, helpers):

    df = helpers.add_column(df_spark, blocking_key, "blocking_key", str)

    df = df.repartition(num_partitions, "blocking_key")

    strategies = {
        "address": (exact(),),
        "email": (exact(),),
    }
    dp = Dedupe(df, spark_session=spark)
    dp.apply(strategies)
    df = dp.canonicalize()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_ids
