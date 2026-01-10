from unittest.mock import ANY, Mock, patch, create_autospec

from pandas.testing import assert_frame_equal
import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, Row, SparkSession
from pyspark.sql.types import StructField, StringType, IntegerType
import pytest

import dupegrouper.base
from dupegrouper.base import (
    Duped,
    _validate_keep_arg,
    _validate_spark_args,
)

import dupegrouper.constants
from dupegrouper.constants import DEFAULT_STRAT_KEY
from dupegrouper.strats_library import BaseProtocol, BaseStrategy
from dupegrouper.dataframe import (
    PandasDF,
    PolarsDF,
    SparkDF,
    SparkRows,
)


##################
# Initialization #
##################


@patch("dupegrouper.base.LocalExecutor")
@patch("dupegrouper.base.SparkExecutor")
@patch("dupegrouper.base.wrap")
def test_init_uses_executor(mock_wrap, mock_spark_executor, mock_local_executor, dataframe):
    df, spark, id = dataframe

    mock_wrap.return_value = Mock()
    dupe = Duped(df, spark_session=spark, id=id, keep="first")
    if spark:
        mock_spark_executor.assert_called_once_with(keep="first", spark_session=spark, id=id)
    else:
        mock_local_executor.assert_called_once_with(keep="first")
    mock_wrap.assert_called_once_with(df, id)
    assert dupe._executor is not None


##############
# Validation #
##############


@pytest.mark.parametrize("keep", ["first", "last"])
def test_validate_keep_arg_valid(keep):
    assert _validate_keep_arg(keep) == keep


def test_validate_keep_arg_invalid():
    with pytest.raises(ValueError, match="Keep must be one of 'first' or 'last'"):
        _validate_keep_arg("middle")


def test_validate_spark_args_valid(mock_spark_session):
    session, id_ = _validate_spark_args(mock_spark_session, "uid")
    assert session == mock_spark_session
    assert id_ == "uid"


def test_validate_spark_args_missing_session():
    with pytest.raises(ValueError, match="spark_session must be provided for a spark dataframe"):
        _validate_spark_args(None, "uid")


def test_validate_spark_args_missing_id(mock_spark_session):
    with pytest.raises(ValueError, match="unique id label must be provided for a spark dataframe"):
        _validate_spark_args(mock_spark_session, None)


##############################
# StrategyManager delegation #
##############################


@patch("dupegrouper.base.StrategyManager")
def test_apply_delegates_to_strategy_manager(mock, dataframe):
    df, spark, id = dataframe

    strat = Mock()
    mock_sm = mock.return_value
    dupe = Duped(df, spark_session=spark, id=id)
    dupe.apply({"address": strat})
    mock_sm.apply.assert_called_once_with({"address": strat})


######################
# canonicalize tests #
######################


@patch("dupegrouper.base.LocalExecutor")
@patch("dupegrouper.base.wrap")
@patch("dupegrouper.base.StrategyManager")
def test_canonicalize_calls_executor_and_resets_strategy_manager(mock_sm, mock_wrap, mock_local, dataframe):

    df, spark, id = dataframe

    strat = Mock()
    
    mock_wrap.return_value = Mock()
    mock_executor = mock_local.return_value
    mock_sm = mock_sm.return_value
    mock_sm.get.return_value = {"address": strat}

    dupe = Duped(df, spark_session=spark, id=id)
    dupe._executor = mock_executor
    dupe._sm = mock_sm

    dupe.canonicalize(columns="address")

    mock_sm.get.assert_called_once()
    mock_executor.canonicalize.assert_called_once_with(mock_wrap.return_value, "address", {"address": strat})
    mock_sm.reset.assert_called_once()


#######################
# Property attributes #
#######################

@patch("dupegrouper.base.StrategyManager")
def test_strats_property_returns_manager_output(mock_sm, dataframe):
    df, spark, id = dataframe
    mock_sm = mock_sm.return_value
    mock_sm.pretty_get.return_value = ("strategy1",)

    dupe = Duped(df, spark_session=spark, id=id)
    dupe._sm = mock_sm
    assert dupe.strats == ("strategy1",)

@patch("dupegrouper.base.wrap")
def test_df_property_returns_unwrapped_df(mock_wrap, df_pandas):
    mock_df_wrapper = Mock()
    mock_df_wrapper.unwrap.return_value = df_pandas
    mock_wrap.return_value = mock_df_wrapper

    dupe = Duped(df_pandas)
    assert_frame_equal(dupe.df, df_pandas)


# ###############################
# # TEST _call_strategy_canonicalizer #
# ###############################


# def test__call_strategy_canonicalizer_deduplication_strategy(duped_mock, strategy_mock):
#     attr = "address"

#     canonicalized_df_mock = Mock()
#     strategy_mock.set_frame.return_value.set_rule.return_value.canonicalize.return_value = canonicalized_df_mock

#     # call

#     result = duped_mock._call_strategy_canonicalizer(strategy_mock, attr)

#     # assert

#     strategy_mock.set_frame.assert_called_once_with(duped_mock._df)
#     strategy_mock.set_frame.return_value.set_rule.return_value.canonicalize.assert_called_once_with(attr)

#     assert result == canonicalized_df_mock

# ################
# # TEST _canonicalize #
# ################


# def test__canonicalize_str_attr(duped_mock, strategy_mock):
#     attr = "address"

#     strategy_collection = {
#         DEFAULT_STRAT_KEY: [
#             strategy_mock,
#             strategy_mock,
#             strategy_mock,
#         ]
#     }

#     with patch.object(duped_mock, "_call_strategy_canonicalizer") as call_canonicalizer:

#         df1 = (Mock(),)  # i.e. after first
#         df2 = (Mock(),)  # ...
#         df3 = (Mock(),)  # after third

#         call_canonicalizer.side_effect = [
#             df1,
#             df2,
#             df3,
#         ]

#         duped_mock._canonicalize(attr, strategy_collection)

#         assert call_canonicalizer.call_count == 3

#         call_canonicalizer.assert_any_call(strategy_mock, attr)
#         call_canonicalizer.assert_any_call(strategy_mock, attr)
#         call_canonicalizer.assert_any_call(strategy_mock, attr)

#         assert duped_mock._df == df3


# def test__canonicalize_nonetype_attr(duped_mock, strategy_mock):

#     attr = None  # Important!

#     strategy_collection = {
#         "attr1": [strategy_mock, strategy_mock],
#         "attr2": [strategy_mock, strategy_mock],
#     }

#     with patch.object(duped_mock, "_call_strategy_canonicalizer") as call_canonicalizer:

#         df1 = (Mock(),)  # i.e. after first
#         df2 = (Mock(),)  # ...
#         df3 = (Mock(),)  # ...
#         df4 = (Mock(),)  # after fourth canonicalize

#         call_canonicalizer.side_effect = [df1, df2, df3, df4]

#         duped_mock._canonicalize(attr, strategy_collection)

#         assert call_canonicalizer.call_count == 4

#         call_canonicalizer.assert_any_call(strategy_mock, "attr1")
#         call_canonicalizer.assert_any_call(strategy_mock, "attr1")
#         call_canonicalizer.assert_any_call(strategy_mock, "attr2")
#         call_canonicalizer.assert_any_call(strategy_mock, "attr2")

#         assert duped_mock._df == df4


# @pytest.mark.parametrize(
#     "attr_input, type",
#     [
#         (42, r".*int.*"),
#         ([42], r".*list.*"),
#         ({"a": 42}, r".*dict.*"),
#         (42.0, r".*float.*"),
#     ],
#     ids=["invalid int", "invalid list", "invalid dict", "invalid float"],
# )
# def test__canonicalize_raises(attr_input, type, duped_mock):
#     with pytest.raises(NotImplementedError, match=f"Unsupported attribute type: {type}"):
#         duped_mock._canonicalize(attr_input, {})  # any dict


# #####################
# # TEST apply #
# #####################


# def test_apply_deduplication_strategy_or_tuple(strategy_mock, duped_mock):
#     with patch.object(duped_mock, "_strategy_manager") as strategy_manager:
#         with patch.object(strategy_manager, "apply") as apply:
#             duped_mock.apply(strategy_mock)
#             assert apply.call_count == 1


# def test_apply_dict(duped_mock, strategy_mock):

#     strategy = {
#         "attr1": [strategy_mock, strategy_mock],
#         "attr2": [strategy_mock, strategy_mock],
#     }

#     with patch.object(duped_mock, "_strategy_manager") as strategy_manager:
#         with patch.object(strategy_manager, "apply") as apply:
#             duped_mock.apply(strategy)
#             assert apply.call_count == 1


# @pytest.mark.parametrize(
#     "strategy, type",
#     [
#         (DummyClass(), r".*DummyClass.*"),
#         ([42], r".*list.*"),
#     ],
#     ids=["invalid class", "invalid list"],
# )
# def test_apply_raises(strategy, type, duped_mock):
#     with pytest.raises(NotImplementedError, match=f"Unsupported strategy: {type}"):
#         duped_mock.apply(strategy)


# ###########################
# # TEST _canonicalize_spark #
# ###########################


# @pytest.fixture
# def mocked_spark_dupegrouper():
#     df_mock = Mock(spec=SparkDataFrame)
#     id_mock = Mock()

#     with patch("dupegrouper.base.wrap"):
#         instance = Duped(df_mock, id_mock)
#         instance._df = df_mock
#         instance._id = "id"
#         instance._id = "id"
#         instance._df.dtypes = [("id", "int"), ("address", "string"), ("email", "string")]
#         instance._df.columns = ["id", "address", "email"]
#         instance._df.schema.fields = [
#             StructField("id", IntegerType()),
#             StructField("address", StringType()),
#             StructField("email", StringType()),
#         ]
#         instance._spark_session = Mock()

#         mock_rdd = Mock()
#         instance._df.rdd = mock_rdd
#         mock_rdd.mapPartitions.return_value = Mock(name="dummy_rdd")

#         instance._mock_rdd = mock_rdd
#         instance._dummy_rdd = mock_rdd.mapPartitions.return_value

#         yield instance


# def test_canonicalize_spark(mocked_spark_dupegrouper, strategy_mock):

#     dg = mocked_spark_dupegrouper

#     attr = "address"
#     strategies = {"address": [strategy_mock, strategy_mock]}

#     mock_df_result = Mock()
#     mock_spark = dg._spark_session
#     mock_spark.createDataFrame.return_value = mock_df_result

#     with patch(
#         "dupegrouper.base._process_partition",
#         return_value=iter([Row(id="1", address="45th street", email="random@ghs.com", canonical_id=1)]),
#     ):
#         with patch("dupegrouper.base.SparkDF") as mockwrapped_df:
#             mockwrapped_result = Mock()
#             mockwrapped_df.return_value = mockwrapped_result

#             dg._canonicalize_spark(attr, strategies)

#             # Assertions
#             dg._mock_rdd.mapPartitions.assert_called_once()
#             mock_spark.createDataFrame.assert_called_once_with(dg._dummy_rdd, schema=ANY)
#             mockwrapped_df.assert_called_once_with(mock_df_result, "id")

#             assert dg._df == mockwrapped_result


# ###########################
# # TEST _process_partition #
# ###########################


# @pytest.fixture
# def partition():
#     return iter(
#         [
#             Row(id=1, address="123 Fake St", email="a@example.com"),
#             Row(id=2, address="123 Fake St", email="another@example.com"),
#         ]
#     )


# def test__process_partition_empty_iter(strategy_mock):
#     strategies = {"address": [strategy_mock]}
#     result = list(_process_partition(iter([]), strategies, id="id", attr="address"))
#     assert result == []


# @patch("dupegrouper.base.Duped")
# def test__process_partition_calls_canonicalize(duped_mock, partition):
#     mock_instance = Mock()
#     mock_instance.df = [Row(id=1, canonical_id=0), Row(id=2, canonical_id=0)]
#     duped_mock.return_value = mock_instance

#     strategy_mock = Mock()
#     strategy_mock.reinstantiate.return_value = strategy_mock
#     strategies = {"address": [strategy_mock]}

#     result = list(_process_partition(partition, strategies, id="id", attr="address"))

#     duped_mock.assert_called_once()
#     mock_instance.apply.assert_called_once_with(strategies)
#     mock_instance.canonicalize.assert_called_once_with("address")
#     assert result == mock_instance.df


# @patch("dupegrouper.base.Duped")
# def test__process_partitions_reinstantiated(duped_mock, partition):
#     mock_instance = Mock()
#     mock_instance.df = [Row(id=1, canonical_id=0)]
#     duped_mock.return_value = mock_instance

#     mock_strategy = Mock()
#     mock_strategy.reinstantiate.return_value = "reinstantiated_strategy"

#     strategies = {"address": [mock_strategy]}

#     result = list(_process_partition(partition, strategies, "id", "address"))

#     mock_strategy.reinstantiate.assert_called_once()
#     assert result == mock_instance.df


# #####################
# # TEST canonicalize #
# #####################


# @pytest.fixture
# def dupgrouper_context(request):
#     df_type, dfwrapper = request.param
#     with patch("dupegrouper.base.wrap"):
#         dg = Duped(Mock(spec=df_type), "id")
#         dg._df = Mock(spec=dfwrapper)

#         with patch.object(dg, "_strategy_manager") as strategy_manager:

#             strategy_manager.get.return_value = {"address": [Mock()]}
#             strategy_manager.reset.return_value = Mock()
#             dg._canonicalize_spark = Mock()
#             dg._canonicalize = Mock()

#             yield {
#                 "dg": dg,
#                 "strategy_manager": strategy_manager,
#                 "is_spark": "SparkDF" == dfwrapper.__name__,
#             }


# @pytest.mark.parametrize(
#     "dupgrouper_context",
#     [
#         (pd.DataFrame, PandasDF),
#         (pl.DataFrame, PolarsDF),
#         (SparkDataFrame, SparkDF),
#         (list[Row], SparkRows),
#     ],
#     indirect=True,
#     ids=["pandas context", "polars context", "spark dataframe context", "spark list rows context"],
# )
# def test_canonicalize(dupgrouper_context):

#     dg = dupgrouper_context["dg"]
#     strategy = dupgrouper_context["strategy_manager"]
#     is_spark = dupgrouper_context["is_spark"]

#     dg.canonicalize("address")

#     if is_spark:
#         dg._canonicalize_spark.assert_called_once_with("address", strategy.get.return_value)
#         dg._canonicalize.assert_not_called()
#     else:
#         dg._canonicalize.assert_called_once_with("address", strategy.get.return_value)
#         dg._canonicalize_spark.assert_not_called()
#     strategy.reset.assert_called_once()


# ##################################
# # TEST Duped - public API! #
# ##################################


# def patch_helper_reset(grouper: Duped):
#     with patch.object(grouper, "_canonicalize") as mock_canonicalize, patch.object(
#         grouper._strategy_manager, "reset"
#     ) as mock_reset:

#         mock_canonicalize.side_effect = mock_reset

#         grouper.canonicalize("address")

#         mock_canonicalize.assert_called_once_with("address", ANY)

#         grouper._strategy_manager = StrategyManager()

#     assert not grouper.strategies


# def test_dupegrouper_strategies_attribute_inline(df_pandas):
#     grouper = Duped(df_pandas)

#     grouper.apply(Exact())
#     grouper.apply(Fuzzy())

#     assert grouper.strategies == ("Exact", "Fuzzy")

#     patch_helper_reset(grouper)


# def test_dupegrouper_strategies_attribute_dict(df_pandas):
#     grouper = Duped(df_pandas)

#     grouper.apply(
#         {
#             "address": [
#                 Exact(),
#             ],
#             "email": [Exact(), Fuzzy()],
#         }
#     )

#     assert grouper.strategies == dict({"address": ("Exact",), "email": ("Exact", "Fuzzy")})

#     patch_helper_reset(grouper)
