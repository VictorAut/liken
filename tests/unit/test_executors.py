from unittest.mock import ANY, Mock, patch

import pytest
from pyspark.rdd import RDD
from pyspark.sql import Row

from enlace._executors import LocalExecutor, SparkExecutor
from enlace._strats_library import BaseStrategy
from enlace._strats_manager import SEQUENTIAL_API_DEFAULT_KEY, StratsDict


############################
# Fixtures
############################


@pytest.fixture
def mock_strategy():
    strat = Mock(spec=BaseStrategy)
    strat.set_frame.return_value = strat
    strat.canonicalizer.return_value = "df_out"
    return strat


@pytest.fixture
def strats_config(mock_strategy):
    cfg = StratsDict({SEQUENTIAL_API_DEFAULT_KEY: [mock_strategy]})
    return cfg


@pytest.fixture
def local_df():
    return Mock()


#################
# LocalExecutor #
#################


def test_localexecutor_canonicalize_sequential_calls(mock_strategy, local_df, strats_config):
    mock_strategy.build_union_find.return_value = ({0: 0}, 1)

    executor = LocalExecutor()

    executor.execute(
        local_df,
        columns="address",
        strats=strats_config,
        keep="last",
        drop_duplicates=False,
        drop_canonical_id=False,
    )

    mock_strategy.set_frame.assert_called_once_with(local_df)
    mock_strategy.canonicalizer.assert_called_once_with(components=ANY, drop_duplicates=False, keep="last")


def test_localexecutor_canonicalize_dict_calls(mock_strategy, local_df):
    mock_strategy.build_union_find.return_value = ({0: 0}, 1)

    cfg = StratsDict({"address": (mock_strategy,), "email": (mock_strategy, mock_strategy)})

    executor = LocalExecutor()

    executor.execute(
        local_df,
        columns=None,
        strats=cfg,
        keep="first",
        drop_duplicates=False,
        drop_canonical_id=False,
    )

    mock_strategy.set_frame.assert_called()
    assert mock_strategy.canonicalizer.call_count == 3


# TODO: test_localexecutor_canonicalize_rules_calls


#################
# SparkExecutor #
#################


def test_sparkexecutor_init_sets_attributes():
    spark = Mock()
    executor = SparkExecutor(spark_session=spark, id="id")

    assert executor._spark_session is spark
    assert executor._id == "id"


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
    strats_config,
):
    spark = Mock()
    executor = SparkExecutor(spark_session=spark, id="id")

    spark_df_result = Mock()
    spark_df_result.select.return_value = spark_df_result
    spark.createDataFrame.return_value = spark_df_result

    executor.execute(
        spark_df,
        columns="address",
        strats=strats_config,
        keep="first",
        drop_duplicates=False,
        drop_canonical_id=False,
    )

    spark_df._df.mapPartitions.assert_called_once()
    spark.createDataFrame.assert_called_once()


######################
# _process_partition #
######################


@patch("enlace.dedupe.Dedupe")
def test_process_partition_empty_partition_returns_empty(mock_dedupe):
    result = list(
        SparkExecutor._process_partition(
            factory=mock_dedupe,
            partition=iter([]),
            strats=Mock(),
            id="id",
            columns="address",
            keep="first",
            drop_duplicates=False,
        )
    )

    assert result == []
    mock_dedupe.assert_not_called()


@patch("enlace.dedupe.Dedupe")
def test_process_partition_calls_dedupe_api(mock_dedupe):
    row = Row(id="1")
    dp_instance = Mock()
    dp_instance.df = ["out"]

    mock_dedupe.return_value = dp_instance

    result = list(
        SparkExecutor._process_partition(
            factory=mock_dedupe,
            partition=iter([row]),
            strats=Mock(),
            id="id",
            columns="address",
            keep="last",
            drop_duplicates=False,
        )
    )

    mock_dedupe.assert_called_once_with([row], id="id")
    dp_instance.apply.assert_called_once()
    dp_instance.canonicalize.assert_called_once_with(
        "address",
        keep="last",
        drop_duplicates=False,
    )
    assert result == ["out"]
