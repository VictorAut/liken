"""Test that canonicalisation is only operated upon per partition"""

from __future__ import annotations

import pytest

from dupegrouper.base import Duped
from dupegrouper.constants import CANONICAL_ID
from dupegrouper.strats_library import exact


@pytest.mark.parametrize(
    "num_partitions, expected_ids",
    [
        (1, [0, 1, 2, 3, 4, 5, 4, 7, 0, 0, 10, 11, 12]),
        (2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 10, 11, 12]),
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
    dg = Duped(df, spark_session=spark)
    dg.apply(strategies)
    dg.canonicalize()

    assert helpers.get_column_as_list(dg.df, CANONICAL_ID) == expected_ids
