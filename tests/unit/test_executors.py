from unittest.mock import Mock, patch, ANY

import pytest
from pyspark.rdd import RDD
from pyspark.sql import Row

from dupegrouper.executors import LocalExecutor, SparkExecutor
from dupegrouper.strats_library import BaseStrategy
from dupegrouper.strats_manager import DEFAULT_STRAT_KEY, StratsConfig

############################
# Fixtures
############################


@pytest.fixture
def mock_strategy():
    strat = Mock(spec=BaseStrategy)
    strat.set_frame.return_value = strat
    strat.set_keep.return_value = strat
    strat.canonicalizer.return_value = "df_out"
    return strat


@pytest.fixture
def strats_config(mock_strategy):
    cfg = StratsConfig({DEFAULT_STRAT_KEY: [mock_strategy]})
    return cfg


@pytest.fixture
def local_df():
    return Mock()


#################
# LocalExecutor #
#################


def test_localexecutor_canonicalize_inline_style_calls(mock_strategy, local_df, strats_config):
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
    mock_strategy.set_keep.assert_called_once_with("last")
    mock_strategy.canonicalizer.assert_called_once_with(components=ANY, drop_duplicates=False)


def test_localexecutor_canonicalize_dict_style_calls(mock_strategy, local_df):
    mock_strategy.build_union_find.return_value = ({0: 0}, 1)

    cfg = StratsConfig({"address": (mock_strategy,), "email": (mock_strategy, mock_strategy)})

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
    mock_strategy.set_keep.assert_called_with("first")
    assert mock_strategy.canonicalizer.call_count == 3


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


@patch("dupegrouper.base.Duped")
def test_process_partition_empty_partition_returns_empty(mock_duped):
    result = list(
        SparkExecutor._process_partition(
            factory=mock_duped,
            partition=iter([]),
            strats=Mock(),
            id="id",
            columns="address",
            keep="first",
            drop_duplicates=False,
        )
    )

    assert result == []
    mock_duped.assert_not_called()


@patch("dupegrouper.base.Duped")
def test_process_partition_calls_duped_api(mock_duped):
    row = Row(id="1")
    dp_instance = Mock()
    dp_instance.df = ["out"]

    mock_duped.return_value = dp_instance

    result = list(
        SparkExecutor._process_partition(
            factory=mock_duped,
            partition=iter([row]),
            strats=Mock(),
            id="id",
            columns="address",
            keep="last",
            drop_duplicates=False,
        )
    )

    mock_duped.assert_called_once_with([row], id="id")
    dp_instance.apply.assert_called_once()
    dp_instance.canonicalize.assert_called_once_with(
        "address",
        keep="last",
        drop_duplicates=False,
    )
    assert result == ["out"]
