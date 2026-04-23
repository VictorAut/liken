from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pyspark.rdd import RDD
from pyspark.sql import Row

from liken._collections import SEQUENTIAL_API_DEFAULT_KEY
from liken._collections import DeduplicationDict
from liken._dedupers import BaseDeduper
from liken.backends.pyspark.executor import SparkExecutor
from liken.backends.ray.executor import RayExecutor
from liken.core.executor import LocalExecutor


############################
# Fixtures
############################


@pytest.fixture
def mock_deduper():
    deduper = Mock(spec=BaseDeduper)
    deduper.set_frame.return_value = deduper
    deduper.canonicalizer.return_value = "df_out"
    return deduper


@pytest.fixture
def dedupers_config(mock_deduper):
    cfg = DeduplicationDict({SEQUENTIAL_API_DEFAULT_KEY: [mock_deduper]})
    return cfg


@pytest.fixture
def local_df():
    return Mock()


#################
# LocalExecutor #
#################


def test_localexecutor_canonicalize_sequential_calls(mock_deduper, local_df, dedupers_config):
    mock_deduper.build_union_find.return_value = ({0: 0}, 1)

    executor = LocalExecutor()

    executor.execute(
        local_df,
        columns="address",
        dedupers=dedupers_config,
        keep="last",
        drop_duplicates=False,
        drop_canonical_id=False,
    )

    mock_deduper.set_frame.assert_called_once_with(local_df)
    mock_deduper.canonicalizer.assert_called_once_with(components=ANY, drop_duplicates=False, keep="last")


def test_localexecutor_canonicalize_dict_calls(mock_deduper, local_df):
    mock_deduper.build_union_find.return_value = ({0: 0}, 1)

    cfg = DeduplicationDict({"address": (mock_deduper,), "email": (mock_deduper, mock_deduper)})

    executor = LocalExecutor()

    executor.execute(
        local_df,
        columns=None,
        dedupers=cfg,
        keep="first",
        drop_duplicates=False,
        drop_canonical_id=False,
    )

    mock_deduper.set_frame.assert_called()
    assert mock_deduper.canonicalizer.call_count == 3


#################
# SparkExecutor #
#################


def test_sparkexecutor_init_sets_attributes():
    spark = Mock()
    executor = SparkExecutor(spark_session=spark)

    assert executor._spark_session is spark


@pytest.fixture
def spark_df():
    df = Mock()

    mock_rdd = Mock(spec=RDD)
    mock_rdd.mapPartitions.return_value = mock_rdd

    df.mapPartitions = Mock(side_effect=mock_rdd.mapPartitions)

    df._df = mock_rdd
    df._schema = Mock()
    return df


def test_sparkexecutor_canonicalize_maps_partitions(
    spark_df,
    dedupers_config,
):
    spark = Mock()
    executor = SparkExecutor(spark_session=spark)

    spark_df_result = Mock()
    spark_df_result.select.return_value = spark_df_result
    spark.createDataFrame.return_value = spark_df_result

    executor.execute(
        spark_df,
        columns="address",
        dedupers=dedupers_config,
        keep="first",
        drop_duplicates=False,
        drop_canonical_id=False,
    )

    spark_df._df.mapPartitions.assert_called_once()
    spark.createDataFrame.assert_called_once()


######################
# _process_partition #
######################


@patch("liken.liken.Dedupe")
def test_process_partition_empty_partition_returns_empty(mock_dedupe):
    result = list(
        SparkExecutor._process_partition(
            factory=mock_dedupe,
            partition=iter([]),
            dedupers=Mock(),
            id="id",
            columns="address",
            keep="first",
            drop_duplicates=False,
        )
    )

    assert result == []
    mock_dedupe.assert_not_called()


@patch("liken.liken.Dedupe")
def test_process_partition_calls_dedupe_api(mock_dedupe):
    row = Row(id="1")
    instance = Mock()
    instance.apply.return_value = instance
    instance.canonicalize.return_value = instance
    instance.collect.return_value = ["out"]

    mock_dedupe._from_rows.return_value = instance

    result = list(
        SparkExecutor._process_partition(
            factory=mock_dedupe,
            partition=iter([row]),
            dedupers=Mock(),
            id="id",
            columns="address",
            keep="last",
            drop_duplicates=False,
        )
    )

    mock_dedupe._from_rows.assert_called_once_with([row])
    instance.apply.assert_called_once()
    instance.canonicalize.assert_called_once_with(
        "address",
        keep="last",
        drop_duplicates=False,
        id="id",
    )
    instance.collect.assert_called_once()
    assert result == ["out"]


#################
# RayExecutor
#################


def test_rayexecutor_init_does_not_crash():
    executor = RayExecutor()
    assert executor is not None
