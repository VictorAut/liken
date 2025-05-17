"""Tests for dupegrouper.base"""

import importlib
import os
from unittest.mock import ANY, Mock, patch

import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame, Row
from pyspark.sql.types import StructField, StringType, IntegerType
import pytest

from dupegrouper.base import (
    DupeGrouper,
    DeduplicationStrategy,
    StrategyTypeError,
    _StrategyManager,
    _wrap,
    _process_partition,
)

import dupegrouper.definitions
from dupegrouper.strategies import Exact, Fuzzy
from dupegrouper.wrappers import WrappedDataFrame
from dupegrouper.wrappers.dataframes import (
    WrappedPandasDataFrame,
    WrappedPolarsDataFrame,
    WrappedSparkDataFrame,
    WrappedSparkRows,
)


# dummy


class DummyClass:
    pass


def dummy_func():
    pass


###############
#  TEST _wrap #
###############

DATAFRAME_TYPES = {
    pd.DataFrame: WrappedPandasDataFrame,
    pl.DataFrame: WrappedPolarsDataFrame,
    SparkDataFrame: WrappedSparkDataFrame,
    list[Row]: WrappedSparkRows,
}


def test__wrap_dataframe(dataframe):
    df, _, id = dataframe

    expected_type = DATAFRAME_TYPES.get(type(df))

    df_wrapped: WrappedDataFrame = _wrap(df, id)

    assert isinstance(df_wrapped, expected_type)


def test__wrap_dataframe_raises():
    with pytest.raises(NotImplementedError, match="Unsupported data frame"):
        _wrap(DummyClass())


######################
#  TEST set group_id #
######################


def reload():
    importlib.reload(dupegrouper.definitions)  # reset constant
    importlib.reload(dupegrouper.wrappers.dataframes._pandas)
    importlib.reload(dupegrouper.wrappers.dataframes._polars)
    importlib.reload(dupegrouper.wrappers.dataframes._spark)


@pytest.mark.parametrize(
    "env_var_value, expected_value",
    [
        # i.e. the default
        ("group_id", "group_id"),
        # null override to default, simulates unset
        (None, "group_id"),
        # arbitrary: different value
        ("beep_boop_id", "beep_boop_id"),
        # arbitrary: supported (but bad!) column naming with whitespace
        ("bad group id", "bad group id"),
    ],
    ids=["default", "null", "default-override", "default-override-bad-format"],
)
def test_group_id_env_var(env_var_value, expected_value, lowlevel_dataframe):
    df, wrapper, id = lowlevel_dataframe

    if env_var_value:
        os.environ["GROUP_ID"] = env_var_value
    else:
        os.environ.pop("GROUP_ID", None)  # remove it if exists

    reload()

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        assert expected_value not in df.columns  # no change
    elif isinstance(df, WrappedSparkRows):
        for row in df.unwrap():
            print(row.asDict().keys())
            assert expected_value in row.asDict().keys()
    else:
        assert expected_value in df.columns

    # clean up
    os.environ["GROUP_ID"] = "group_id"

    reload()


##############################################
#  TEST _StrategyManager + StrategyTypeError #
##############################################


DEFAULT_ERROR_MSG = "Input is not valid"
CLASS_ERROR_MSG = "Input class is not valid: must be an instance of `DeduplicationStrategy`"
TUPLE_ERROR_MSG = "Input tuple is not valid: must be a length 2 [callable, dict]"
DICT_ERROR_MSG = "Input dict is not valid: items must be a list of `DeduplicationStrategy` or tuples"


@pytest.mark.parametrize(
    "strategy, expected_to_pass, base_msg",
    [
        # correct base inputs
        (Mock(spec=DeduplicationStrategy), True, None),
        ((lambda x: x, {"key": "value"}), True, None),
        (
            {
                "address": [
                    Mock(spec=DeduplicationStrategy),
                    (lambda x: x, {"key": "value"}),
                ],
                "email": [
                    Mock(spec=DeduplicationStrategy),
                    Mock(spec=DeduplicationStrategy),
                ],
            },
            True,
            None,
        ),
        # incorrect inputs
        (DummyClass, False, CLASS_ERROR_MSG),
        (lambda x: x, False, DEFAULT_ERROR_MSG),
        ((lambda x: x, [1, 2, 3]), False, TUPLE_ERROR_MSG),
        (("foo",), False, TUPLE_ERROR_MSG),
        (["bar", "baz"], False, DEFAULT_ERROR_MSG),
        ("foobar", False, DEFAULT_ERROR_MSG),
        (
            {
                "address": [DummyClass()],
                "email": [
                    "random string",
                    ("tuple too short",),
                ],
            },
            False,
            DICT_ERROR_MSG,
        ),
    ],
    ids=[
        "valid dedupe class",
        "valid callable",
        "valid dict",
        "invalid class",
        "invalid callable not in tuple",
        "invalid callable positional args",
        "invalid tuple",
        "invalid list",
        "invalid str",
        "invalid dict",
    ],
)
def test__strategy_manager_validate_addition_strategy(strategy, expected_to_pass, base_msg):
    """validates that the input 'strtagey' is legit, against `StrategyTypeError`"""
    manager = _StrategyManager()
    if expected_to_pass:
        if isinstance(strategy, dict):
            for k, value in strategy.items():
                for v in value:
                    manager.add(k, v)
                    assert (k in manager.get()) is expected_to_pass
        else:
            manager.add("default", strategy)
            assert ("default" in manager.get()) is expected_to_pass
    else:
        with pytest.raises(StrategyTypeError) as e:
            manager.add("default", strategy)
            assert base_msg in str(e)


def test__strategy_manager_reset():
    manager = _StrategyManager()
    strategy = Mock(spec=DeduplicationStrategy)
    manager.add("name", strategy)
    manager.reset()
    assert manager.get() == {}


###############################
# TEST _call_strategy_deduper #
###############################


def test__call_strategy_deduper_deduplication_strategy(dupegrouper_mock, strategy_mock):
    attr = "address"

    deduped_df_mock = Mock()
    strategy_mock.with_frame.return_value.dedupe.return_value = deduped_df_mock

    # call

    result = dupegrouper_mock._call_strategy_deduper(strategy_mock, attr)

    # assert

    strategy_mock.with_frame.assert_called_once_with(dupegrouper_mock._df)
    strategy_mock.with_frame.return_value.dedupe.assert_called_once_with(attr)

    assert result == deduped_df_mock


def test__call_strategy_deduper_tuple(dupegrouper_mock):
    attr = "address"

    mock_callable = Mock()
    mock_callable.__name__ = "mock_func"

    mock_kwargs = {"tolerance": 0.8}

    deduped_df_mock = Mock()

    with patch("dupegrouper.base.Custom") as Custom:

        # Mock instance that Custom returns
        instance = Mock()
        Custom.return_value = instance

        # Ensure full method chain is mocked
        instance.with_frame.return_value = instance
        instance.dedupe.return_value = deduped_df_mock

        result = dupegrouper_mock._call_strategy_deduper(
            (mock_callable, mock_kwargs),  # tuple!
            attr,
        )

        # assert

        Custom.assert_called_once_with(mock_callable, attr, **mock_kwargs)
        instance.with_frame.assert_called_once_with(dupegrouper_mock._df)
        instance.with_frame.return_value.dedupe.assert_called_once_with()

        assert result == deduped_df_mock


@pytest.mark.parametrize(
    "input, type",
    [
        (42, r".*int.*"),
        (DummyClass(), r".*DummyClass.*"),
        (["a"], r".*list.*"),
        ({"a": "b"}, r".*dict.*"),
    ],
    ids=["invalid int", "invalid class", "invalid list", "invalid dict"],
)
def test__call_strategy_deduper_raises(input, type, dupegrouper_mock):
    with pytest.raises(NotImplementedError, match=f"Unsupported strategy: {type}"):
        dupegrouper_mock._call_strategy_deduper(input, "address")


################
# TEST _dedupe #
################


def test__dedupe_str_attr(dupegrouper_mock, strategy_mock):
    attr = "address"

    strategy_collection = {
        "default": [
            strategy_mock,
            strategy_mock,
            strategy_mock,
        ]
    }

    with patch.object(dupegrouper_mock, "_call_strategy_deduper") as call_deduper:

        df1 = (Mock(),)  # i.e. after first
        df2 = (Mock(),)  # ...
        df3 = (Mock(),)  # after third

        call_deduper.side_effect = [
            df1,
            df2,
            df3,
        ]

        dupegrouper_mock._dedupe(attr, strategy_collection)

        assert call_deduper.call_count == 3

        call_deduper.assert_any_call(strategy_mock, attr)
        call_deduper.assert_any_call(strategy_mock, attr)
        call_deduper.assert_any_call(strategy_mock, attr)

        assert dupegrouper_mock._df == df3


def test__dedupe_nonetype_attr(dupegrouper_mock, strategy_mock):

    attr = None  # Important!

    strategy_collection = {
        "attr1": [strategy_mock, strategy_mock],
        "attr2": [strategy_mock, strategy_mock],
    }

    with patch.object(dupegrouper_mock, "_call_strategy_deduper") as call_deduper:

        df1 = (Mock(),)  # i.e. after first
        df2 = (Mock(),)  # ...
        df3 = (Mock(),)  # ...
        df4 = (Mock(),)  # after fourth dedupe

        call_deduper.side_effect = [df1, df2, df3, df4]

        dupegrouper_mock._dedupe(attr, strategy_collection)

        assert call_deduper.call_count == 4

        call_deduper.assert_any_call(strategy_mock, "attr1")
        call_deduper.assert_any_call(strategy_mock, "attr1")
        call_deduper.assert_any_call(strategy_mock, "attr2")
        call_deduper.assert_any_call(strategy_mock, "attr2")

        assert dupegrouper_mock._df == df4


@pytest.mark.parametrize(
    "attr_input, type",
    [
        (42, r".*int.*"),
        ([42], r".*list.*"),
        ((42,), r".*tuple.*"),
        ({"a": 42}, r".*dict.*"),
        (42.0, r".*float.*"),
    ],
    ids=["invalid int", "invalid list", "invalid tuple", "invalid dict", "invalid float"],
)
def test__dedupe_raises(attr_input, type, dupegrouper_mock):
    with pytest.raises(NotImplementedError, match=f"Unsupported attribute type: {type}"):
        dupegrouper_mock._dedupe(attr_input, {})  # any dict


#####################
# TEST add_strategy #
#####################


@pytest.mark.parametrize(
    "strategy",
    [(dummy_func, {"tolerance": 0.8}), Mock(spec=DeduplicationStrategy)],
    ids=["tuple", "DeduplicationStrategy"],
)
def test_add_strategy_deduplication_strategy_or_tuple(strategy, dupegrouper_mock):

    with patch.object(dupegrouper_mock, "_strategy_manager") as strategy_manager:

        with patch.object(strategy_manager, "add") as add:

            dupegrouper_mock.add_strategy(strategy)

            assert add.call_count == 1

            add.assert_any_call("default", strategy)


def test_add_strategy_dict(dupegrouper_mock, strategy_mock):

    strategy = {
        "attr1": [strategy_mock, strategy_mock],
        "attr2": [strategy_mock, strategy_mock],
    }

    with patch.object(dupegrouper_mock, "_strategy_manager") as strategy_manager:

        with patch.object(strategy_manager, "add") as add:

            dupegrouper_mock.add_strategy(strategy)

            assert add.call_count == 4

            add.assert_any_call("attr1", strategy_mock)
            add.assert_any_call("attr1", strategy_mock)
            add.assert_any_call("attr2", strategy_mock)
            add.assert_any_call("attr2", strategy_mock)


@pytest.mark.parametrize(
    "strategy, type",
    [
        (DummyClass(), r".*DummyClass.*"),
        ([42], r".*list.*"),
    ],
    ids=["invalid class", "invalid list"],
)
def test_add_strategy_raises(strategy, type, dupegrouper_mock):
    with pytest.raises(NotImplementedError, match=f"Unsupported strategy: {type}"):
        dupegrouper_mock.add_strategy(strategy)


###########################
# TEST _dedupe_spark #
###########################


@pytest.fixture
def mocked_spark_dupegrouper():
    df_mock = Mock(spec=SparkDataFrame)
    id_mock = Mock()

    with patch("dupegrouper.base._wrap"):
        instance = DupeGrouper(df_mock, id_mock)
        instance._df = df_mock
        instance._id = "id"
        instance._id = "id"
        instance._df.dtypes = [("id", "int"), ("address", "string"), ("email", "string")]
        instance._df.columns = ["id", "address", "email"]
        instance._df.schema.fields = [
            StructField("id", IntegerType()),
            StructField("address", StringType()),
            StructField("email", StringType()),
        ]
        instance._spark_session = Mock()

        mock_rdd = Mock()
        instance._df.rdd = mock_rdd
        mock_rdd.mapPartitions.return_value = Mock(name="dummy_rdd")

        instance._mock_rdd = mock_rdd
        instance._dummy_rdd = mock_rdd.mapPartitions.return_value

        yield instance


def test_dedupe_spark(mocked_spark_dupegrouper, strategy_mock):

    dg = mocked_spark_dupegrouper

    attr = "address"
    strategies = {"address": [strategy_mock, strategy_mock]}

    mock_df_result = Mock()
    mock_spark = dg._spark_session
    mock_spark.createDataFrame.return_value = mock_df_result

    with patch(
        "dupegrouper.base._process_partition",
        return_value=iter([Row(id="1", address="45th street", email="random@ghs.com", group_id=1)]),
    ):
        with patch("dupegrouper.base.WrappedSparkDataFrame") as mock_wrapped_df:
            mock_wrapped_result = Mock()
            mock_wrapped_df.return_value = mock_wrapped_result

            dg._dedupe_spark(attr, strategies)

            # Assertions
            dg._mock_rdd.mapPartitions.assert_called_once()
            mock_spark.createDataFrame.assert_called_once_with(dg._dummy_rdd, schema=ANY)
            mock_wrapped_df.assert_called_once_with(mock_df_result, "id")

            assert dg._df == mock_wrapped_result


###########################
# TEST _process_partition #
###########################


@pytest.fixture
def partition():
    return iter(
        [
            Row(id=1, address="123 Fake St", email="a@example.com"),
            Row(id=2, address="123 Fake St", email="another@example.com"),
        ]
    )


def test__process_partition_empty_iter(strategy_mock):
    strategies = {"address": [strategy_mock]}
    result = list(_process_partition(iter([]), strategies, id="id", attr="address"))
    assert result == []


@patch("dupegrouper.base.DupeGrouper")
def test__process_partition_calls_dedupe(dupegrouper_mock, partition):
    mock_instance = Mock()
    mock_instance.df = [Row(id=1, group_id=0), Row(id=2, group_id=0)]
    dupegrouper_mock.return_value = mock_instance

    strategy_mock = Mock()
    strategy_mock.reinstantiate.return_value = strategy_mock
    strategies = {"address": [strategy_mock]}

    result = list(_process_partition(partition, strategies, id="id", attr="address"))

    dupegrouper_mock.assert_called_once()
    mock_instance.add_strategy.assert_called_once_with(strategies)
    mock_instance.dedupe.assert_called_once_with("address")
    assert result == mock_instance.df


@patch("dupegrouper.base.DupeGrouper")
def test__process_partitions_reinstantiated(dupegrouper_mock, partition):
    mock_instance = Mock()
    mock_instance.df = [Row(id=1, group_id=0)]
    dupegrouper_mock.return_value = mock_instance

    mock_strategy = Mock()
    mock_strategy.reinstantiate.return_value = "reinstantiated_strategy"

    strategies = {"address": [mock_strategy]}

    result = list(_process_partition(partition, strategies, "id", "address"))

    mock_strategy.reinstantiate.assert_called_once()
    assert result == mock_instance.df


#####################
# TEST dedupe #
#####################


@pytest.fixture
def dupgrouper_context(request):
    df_type, df_wrapper = request.param
    with patch("dupegrouper.base._wrap"):
        dg = DupeGrouper(Mock(spec=df_type), "id")
        dg._df = Mock(spec=df_wrapper)

        with patch.object(dg, "_strategy_manager") as strategy_manager:

            strategy_manager.get.return_value = {"address": [Mock()]}
            strategy_manager.reset.return_value = Mock()
            dg._dedupe_spark = Mock()
            dg._dedupe = Mock()

            yield {
                "dg": dg,
                "strategy_manager": strategy_manager,
                "is_spark": "WrappedSparkDataFrame" == df_wrapper.__name__,
            }


@pytest.mark.parametrize(
    "dupgrouper_context",
    [
        (pd.DataFrame, WrappedPandasDataFrame),
        (pl.DataFrame, WrappedPolarsDataFrame),
        (SparkDataFrame, WrappedSparkDataFrame),
        (list[Row], WrappedSparkRows),
    ],
    indirect=True,
    ids=["pandas context", "polars context", "spark dataframe context", "spark list rows context"],
)
def test_dedupe(dupgrouper_context):

    dg = dupgrouper_context["dg"]
    strategy = dupgrouper_context["strategy_manager"]
    is_spark = dupgrouper_context["is_spark"]

    dg.dedupe("address")

    if is_spark:
        dg._dedupe_spark.assert_called_once_with("address", strategy.get.return_value)
        dg._dedupe.assert_not_called()
    else:
        dg._dedupe.assert_called_once_with("address", strategy.get.return_value)
        dg._dedupe_spark.assert_not_called()
    strategy.reset.assert_called_once()


##################################
# TEST DupeGrouper - public API! #
##################################


def patch_helper_reset(grouper: DupeGrouper):
    with patch.object(grouper, "_dedupe") as mock_dedupe, patch.object(
        grouper._strategy_manager, "reset"
    ) as mock_reset:

        mock_dedupe.side_effect = mock_reset

        grouper.dedupe("address")

        mock_dedupe.assert_called_once_with("address", ANY)

        grouper._strategy_manager = _StrategyManager()
        print(grouper.strategies)

    assert not grouper.strategies


def test_dupegrouper_strategies_attribute_inline(df_pandas):
    grouper = DupeGrouper(df_pandas)

    grouper.add_strategy(Mock(spec=Exact))
    grouper.add_strategy(Mock(spec=Fuzzy))
    grouper.add_strategy((dummy_func, {"str": "random"}))

    assert grouper.strategies == tuple(["Exact", "Fuzzy", "dummy_func"])

    patch_helper_reset(grouper)


def test_dupegrouper_strategies_attribute_dict(df_pandas):
    grouper = DupeGrouper(df_pandas)

    grouper.add_strategy(
        {
            "address": [
                Mock(spec=Exact),
                (dummy_func, {"key": "value"}),
            ],
            "email": [
                Mock(spec=Exact),
                Mock(spec=Fuzzy),
            ],
        }
    )

    assert grouper.strategies == dict({"address": ("Exact", "dummy_func"), "email": ("Exact", "Fuzzy")})

    patch_helper_reset(grouper)
