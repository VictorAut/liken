"""Test that canonicalisation is only operated upon per partition"""

from __future__ import annotations

import pytest

import liken as lk
from liken._constants import CANONICAL_ID


PARTITION_1_PARAMS = (1, [1, 2, 3, 4, 5, 5, 7, 1, 5, 10])
PARTITION_2_PARAMS = (2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
PARAMS = [PARTITION_1_PARAMS, PARTITION_2_PARAMS]



IDS=["1 partitions", "2 partitions"]

@pytest.mark.parametrize(
    "num_partitions, expected_ids",
    PARAMS,
    ids=IDS,
)
def test_matrix_spark(num_partitions, expected_ids, df_spark, spark, blocking_key, helpers):

    df = helpers.add_column(df_spark, blocking_key, "blocking_key", str)

    df = df.repartition(num_partitions, "blocking_key")

    dedupers = {
        "address": (lk.exact(),),
        "email": (lk.exact(),),
    }
    df = lk.dedupe(df, spark_session=spark).apply(dedupers).canonicalize(id="id").collect()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_ids

@pytest.mark.parametrize(
    "num_partitions, expected_ids",
    PARAMS,
    ids=IDS,
)
def test_matrix_ray(num_partitions, expected_ids, df_ray, blocking_key, helpers):

    df = helpers.add_column(df_ray, blocking_key, "blocking_key", str)

    df = df.repartition(num_partitions, keys="blocking_key")

    dedupers = {
        "address": (lk.exact(),),
        "email": (lk.exact(),),
    }

    # IMPORTANT!!!: define an id
    df = lk.dedupe(df).apply(dedupers).canonicalize(id="id").collect()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_ids
