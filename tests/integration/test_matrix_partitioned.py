"""Test that canonicalisation is only operated upon per partition"""

from __future__ import annotations

import hashlib

import pytest

import liken as lk
from liken.constants import CANONICAL_ID


PARTITION_1_PARAMS = (1, [1, 2, 3, 4, 5, 5, 7, 1, 5, 10])
PARTITION_2_PARAMS = (2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
PARAMS = [PARTITION_1_PARAMS, PARTITION_2_PARAMS]
DEDUPERS = {
    "address": (lk.exact(),),
    "email": (lk.exact(),),
}

IDS = ["1 partitions", "2 partitions"]


@pytest.mark.parametrize(
    "num_partitions, expected_ids",
    PARAMS,
    ids=IDS,
)
def test_matrix_spark(
    num_partitions,
    expected_ids,
    dataframe,
    spark_session,
    blocking_key,
    helpers,
    request,
):
    backend = request.config.getoption("--backend")

    if backend != "pyspark":
        pytest.skip("Pyspark only test")

    df = helpers.add_column(dataframe, blocking_key, "blocking_key", str)

    df = df.repartition(num_partitions, "blocking_key")

    df = (
        lk.dedupe(df, spark_session=spark_session)
        .apply(DEDUPERS)
        .canonicalize(id="id")
        .collect()
    )

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_ids


@pytest.mark.parametrize(
    "num_partitions, expected_ids",
    PARAMS,
    ids=IDS,
)
def test_matrix_ray(
    num_partitions,
    expected_ids,
    dataframe,
    blocking_key,
    helpers,
    request,
):
    backend = request.config.getoption("--backend")

    if backend != "ray":
        pytest.skip("Ray only test")

    df = helpers.add_column(dataframe, blocking_key, "blocking_key", str)

    df = df.repartition(num_partitions, keys="blocking_key")

    df = lk.dedupe(df).apply(DEDUPERS).canonicalize(id="id").collect()

    assert helpers.get_column_as_list(df, CANONICAL_ID) == expected_ids


@pytest.mark.parametrize(
    "num_partitions, expected_ids",
    PARAMS,
    ids=IDS,
)
def test_matrix_dask(
    num_partitions,
    expected_ids,
    dataframe,
    blocking_key,
    helpers,
    request,
):
    backend = request.config.getoption("--backend")

    if backend != "dask":
        pytest.skip("Dask only test")

    df = helpers.add_column(dataframe, blocking_key, "blocking_key", str)

    def _add_partition(df, npartitions):

        def stable_hash(x):
            return int(hashlib.md5(str(x).encode()).hexdigest(), 16) % npartitions

        df["_part"] = df["blocking_key"].map(stable_hash)
        return df

    meta = df._meta.assign(_part="int64")

    df = df.map_partitions(_add_partition, num_partitions, meta=meta)

    df = df.shuffle("_part", npartitions=num_partitions)

    df = df.drop(columns="_part")

    df = lk.dedupe(df).apply(DEDUPERS).canonicalize(id="id").collect()

    result = helpers.get_column_as_list(df, CANONICAL_ID)

    assert sorted(result) == sorted(expected_ids)
