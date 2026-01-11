import pytest
from unittest.mock import Mock, patch

from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType

from dupegrouper.executors import LocalExecutor, SparkExecutor
from dupegrouper.strats_library import BaseStrategy
from dupegrouper.strats_manager import StratsConfig, DEFAULT_STRAT_KEY
from dupegrouper.constants import CANONICAL_ID


############################
# Fixtures
############################


@pytest.fixture
def mock_strategy():
    strat = Mock(spec=BaseStrategy)
    strat.set_frame.return_value = strat
    strat.set_keep.return_value = strat
    strat.canonicalize.return_value = "df_out"
    return strat


@pytest.fixture
def strats_config(mock_strategy):
    cfg = StratsConfig({DEFAULT_STRAT_KEY: [mock_strategy]})
    return cfg


@pytest.fixture
def local_df():
    return Mock()


@pytest.fixture
def spark_df():
    df = Mock()
    df.rdd = Mock()
    df.schema = StructType([StructField("id", StringType(), True)])
    df.columns = ["id"]
    df.dtypes = [("id", "string")]
    return df


#################
# LocalExecutor #
#################


def test_localexecutor_canonicalize_inline_style_calls(mock_strategy, local_df, strats_config):
    executor = LocalExecutor(keep="last")

    executor.canonicalize(
        df=local_df,
        columns="address",
        strats=strats_config,
    )

    mock_strategy.set_frame.assert_called_once_with(local_df)
    mock_strategy.set_keep.assert_called_once_with("last")
    mock_strategy.canonicalize.assert_called_once_with("address")


def test_localexecutor_canonicalize_dict_style_calls(mock_strategy, local_df):

    cfg = StratsConfig({"address": (mock_strategy,), "email": (mock_strategy, mock_strategy)})

    executor = LocalExecutor(keep="first")

    executor.canonicalize(
        df=local_df,
        columns=None,
        strats=cfg,
    )

    mock_strategy.set_frame.assert_called()
    mock_strategy.set_keep.assert_called_with("first")
    assert mock_strategy.canonicalize.call_count == 3


#################
# SparkExecutor #
#################


def test_sparkexecutor_init_sets_attributes():
    spark = Mock()
    executor = SparkExecutor(keep="first", spark_session=spark, id="id")

    assert executor._keep == "first"
    assert executor._spark_session is spark
    assert executor._id == "id"


@patch("dupegrouper.executors.SparkExecutor._get_schema")
@patch("dupegrouper.executors.SparkExecutor._add_canonical_id")
def test_sparkexecutor_canonicalize_maps_partitions(
    mock_add_canonical_id,
    mock_get_schema,
    spark_df,
    strats_config,
):
    spark = Mock()
    executor = SparkExecutor(keep="first", spark_session=spark, id="id")

    # mock RDD pipeline
    mock_rdd = Mock()
    mock_rdd.mapPartitions.return_value = mock_rdd
    mock_add_canonical_id.return_value = mock_rdd

    # mock schema handling
    mock_get_schema.return_value = Mock()

    # mock Spark createDataFrame
    spark.createDataFrame.return_value = "new_df"

    executor.canonicalize(
        df=spark_df,
        columns="address",
        strats=strats_config,
    )

    # assertions that actually matter
    mock_add_canonical_id.assert_called_once()
    mock_rdd.mapPartitions.assert_called_once()
    spark.createDataFrame.assert_called_once()


def test_sparkexecutor_get_schema_adds_canonical_id():
    spark = Mock()
    executor = SparkExecutor(keep="first", spark_session=spark, id="id")

    df = Mock()
    df.schema.fields = [StructField("id", StringType(), True)]
    df.columns = ["id"]
    df.dtypes = [("id", "string")]

    schema = executor._get_schema(df)

    field_names = {f.name for f in schema.fields}
    assert CANONICAL_ID in field_names


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
        )
    )

    mock_duped.assert_called_once_with([row], id="id", keep="last")
    dp_instance.apply.assert_called_once()
    dp_instance.canonicalize.assert_called_once_with("address")
    assert result == ["out"]
