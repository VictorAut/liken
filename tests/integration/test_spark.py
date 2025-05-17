from __future__ import annotations

import pytest

from dupegrouper.base import DupeGrouper
from dupegrouper.definitions import GROUP_ID
from dupegrouper.strategies import Exact


@pytest.mark.parametrize(
    "num_partitions, expected_ids",
    [(1, [1, 2, 3, 4, 5, 6, 5, 8, 1, 1, 11, 12, 13]), (2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 13])],
    ids=["1 partitions", "2 partitions"],
)
def test_spark_partitions(num_partitions, expected_ids, df_spark, spark, helpers):

    df_spark = df_spark.repartition(num_partitions, "blocking_key")

    strategies = {
        "address": [Exact()],
        "email": [Exact()],
    }
    dg = DupeGrouper(df_spark, spark_session=spark, id="id")
    dg.add_strategy(strategies)
    dg.dedupe()

    assert helpers.get_column_as_list(dg.df, GROUP_ID) == expected_ids
