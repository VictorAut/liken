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
    Duped,
    BaseStrategy,
    StrategyTypeError,
    _StrategyManager,
    wrap,
    _process_partition,
)

import dupegrouper.definitions
from dupegrouper.strategies import Exact, Fuzzy
from dupegrouper.dataframe import (
    WrappedDataFrame,
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
#  TEST wrap #
###############

DATAFRAME_TYPES = {
    pd.DataFrame: WrappedPandasDataFrame,
    pl.DataFrame: WrappedPolarsDataFrame,
    SparkDataFrame: WrappedSparkDataFrame,
    list[Row]: WrappedSparkRows,
}


def test_wrap_dataframe(dataframe):
    df, _, id = dataframe

    expected_type = DATAFRAME_TYPES.get(type(df))

    wrapped_df: WrappedDataFrame = wrap(df, id)

    assert isinstance(wrapped_df, expected_type)


def test_wrap_dataframe_raises():
    with pytest.raises(NotImplementedError, match="Unsupported data frame"):
        wrap(DummyClass())


######################
#  TEST set canonical_id #
######################


def reload_imports():
    importlib.reload(dupegrouper.definitions)
    importlib.reload(dupegrouper.dataframe)


@pytest.mark.parametrize(
    "env_var_value, expected_value",
    [
        # default
        ("canonical_id", "canonical_id"),
        # override to default
        (None, "canonical_id"),
        # arbitrary: different value
        ("random_id", "random_id"),
    ]
)
def test_canonical_id_env_var(env_var_value, expected_value, lowlevel_dataframe):
    df, wrapper, id = lowlevel_dataframe

    if env_var_value:
        os.environ["CANONICAL_ID"] = env_var_value
    else:
        os.environ.pop("CANONICAL_ID", None)

    reload_imports()

    df = wrapper(df, id)

    if isinstance(df, WrappedSparkDataFrame):
        assert expected_value not in df.columns
    elif isinstance(df, WrappedSparkRows):
        for row in df.unwrap():
            assert expected_value in row.asDict().keys()
    else:
        assert expected_value in df.columns

    os.environ["CANONICAL_ID"] = "canonical_id"

    reload_imports()


##############################################
#  TEST _StrategyManager + StrategyTypeError #
##############################################


DEFAULT_ERROR_MSG = "Input is not valid"
CLASS_ERROR_MSG = "Input class is not valid: must be an instance of `BaseStrategy`"
TUPLE_ERROR_MSG = "Input tuple is not valid: must be a length 2 [callable, dict]"
DICT_ERROR_MSG = "Input dict is not valid: items must be a list of `BaseStrategy` or tuples"


@pytest.mark.parametrize(
    "strategy, expected_to_pass, base_msg",
    [
        # correct base inputs
        (Mock(spec=BaseStrategy), True, None),
        ((lambda x: x, {"key": "value"}), True, None),
        (
            {
                "address": [
                    Mock(spec=BaseStrategy),
                    (lambda x: x, {"key": "value"}),
                ],
                "email": [
                    Mock(spec=BaseStrategy),
                    Mock(spec=BaseStrategy),
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
        "valid canonicalize class",
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
    strategy = Mock(spec=BaseStrategy)
    manager.add("name", strategy)
    manager.reset()
    assert manager.get() == {}


###############################
# TEST _call_strategy_canonicalizer #
###############################


def test__call_strategy_canonicalizer_deduplication_strategy(dupegrouper_mock, strategy_mock):
    attr = "address"

    canonicalized_df_mock = Mock()
    strategy_mock.bind_frame.return_value.bind_rule.return_value.canonicalize.return_value = canonicalized_df_mock

    # call

    result = dupegrouper_mock._call_strategy_canonicalizer(strategy_mock, attr)

    # assert

    strategy_mock.bind_frame.assert_called_once_with(dupegrouper_mock._df)
    strategy_mock.bind_frame.return_value.bind_rule.return_value.canonicalize.assert_called_once_with(attr)

    assert result == canonicalized_df_mock


def test__call_strategy_canonicalizer_tuple(dupegrouper_mock):
    attr = "address"

    mock_callable = Mock()
    mock_callable.__name__ = "mock_func"

    mock_kwargs = {"tolerance": 0.8}

    canonicalized_df_mock = Mock()

    with patch("dupegrouper.base.Custom") as Custom:

        # Mock instance that Custom returns
        instance = Mock()
        Custom.return_value = instance

        # Ensure full method chain is mocked
        instance.bind_frame.return_value.bind_rule.return_value = instance
        instance.canonicalize.return_value = canonicalized_df_mock

        result = dupegrouper_mock._call_strategy_canonicalizer(
            (mock_callable, mock_kwargs),  # tuple!
            attr,
        )

        # assert

        Custom.assert_called_once_with(mock_callable, attr, **mock_kwargs)
        instance.bind_frame.assert_called_once_with(dupegrouper_mock._df)
        instance.bind_frame.return_value.bind_rule.return_value.canonicalize.assert_called_once_with()

        assert result == canonicalized_df_mock


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
def test__call_strategy_canonicalizer_raises(input, type, dupegrouper_mock):
    with pytest.raises(NotImplementedError, match=f"Unsupported strategy: {type}"):
        dupegrouper_mock._call_strategy_canonicalizer(input, "address")


################
# TEST _canonicalize #
################


def test__canonicalize_str_attr(dupegrouper_mock, strategy_mock):
    attr = "address"

    strategy_collection = {
        "default": [
            strategy_mock,
            strategy_mock,
            strategy_mock,
        ]
    }

    with patch.object(dupegrouper_mock, "_call_strategy_canonicalizer") as call_canonicalizer:

        df1 = (Mock(),)  # i.e. after first
        df2 = (Mock(),)  # ...
        df3 = (Mock(),)  # after third

        call_canonicalizer.side_effect = [
            df1,
            df2,
            df3,
        ]

        dupegrouper_mock._canonicalize(attr, strategy_collection)

        assert call_canonicalizer.call_count == 3

        call_canonicalizer.assert_any_call(strategy_mock, attr)
        call_canonicalizer.assert_any_call(strategy_mock, attr)
        call_canonicalizer.assert_any_call(strategy_mock, attr)

        assert dupegrouper_mock._df == df3


def test__canonicalize_nonetype_attr(dupegrouper_mock, strategy_mock):

    attr = None  # Important!

    strategy_collection = {
        "attr1": [strategy_mock, strategy_mock],
        "attr2": [strategy_mock, strategy_mock],
    }

    with patch.object(dupegrouper_mock, "_call_strategy_canonicalizer") as call_canonicalizer:

        df1 = (Mock(),)  # i.e. after first
        df2 = (Mock(),)  # ...
        df3 = (Mock(),)  # ...
        df4 = (Mock(),)  # after fourth canonicalize

        call_canonicalizer.side_effect = [df1, df2, df3, df4]

        dupegrouper_mock._canonicalize(attr, strategy_collection)

        assert call_canonicalizer.call_count == 4

        call_canonicalizer.assert_any_call(strategy_mock, "attr1")
        call_canonicalizer.assert_any_call(strategy_mock, "attr1")
        call_canonicalizer.assert_any_call(strategy_mock, "attr2")
        call_canonicalizer.assert_any_call(strategy_mock, "attr2")

        assert dupegrouper_mock._df == df4


@pytest.mark.parametrize(
    "attr_input, type",
    [
        (42, r".*int.*"),
        ([42], r".*list.*"),
        ({"a": 42}, r".*dict.*"),
        (42.0, r".*float.*"),
    ],
    ids=["invalid int", "invalid list", "invalid dict", "invalid float"],
)
def test__canonicalize_raises(attr_input, type, dupegrouper_mock):
    with pytest.raises(NotImplementedError, match=f"Unsupported attribute type: {type}"):
        dupegrouper_mock._canonicalize(attr_input, {})  # any dict


#####################
# TEST apply #
#####################


@pytest.mark.parametrize(
    "strategy",
    [(dummy_func, {"tolerance": 0.8}), Mock(spec=BaseStrategy)],
    ids=["tuple", "BaseStrategy"],
)
def test_apply_deduplication_strategy_or_tuple(strategy, dupegrouper_mock):

    with patch.object(dupegrouper_mock, "_strategy_manager") as strategy_manager:

        with patch.object(strategy_manager, "add") as add:

            dupegrouper_mock.apply(strategy)

            assert add.call_count == 1

            add.assert_any_call("default", strategy)


def test_apply_dict(dupegrouper_mock, strategy_mock):

    strategy = {
        "attr1": [strategy_mock, strategy_mock],
        "attr2": [strategy_mock, strategy_mock],
    }

    with patch.object(dupegrouper_mock, "_strategy_manager") as strategy_manager:

        with patch.object(strategy_manager, "add") as add:

            dupegrouper_mock.apply(strategy)

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
def test_apply_raises(strategy, type, dupegrouper_mock):
    with pytest.raises(NotImplementedError, match=f"Unsupported strategy: {type}"):
        dupegrouper_mock.apply(strategy)


###########################
# TEST _canonicalize_spark #
###########################


@pytest.fixture
def mocked_spark_dupegrouper():
    df_mock = Mock(spec=SparkDataFrame)
    id_mock = Mock()

    with patch("dupegrouper.base.wrap"):
        instance = Duped(df_mock, id_mock)
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


def test_canonicalize_spark(mocked_spark_dupegrouper, strategy_mock):

    dg = mocked_spark_dupegrouper

    attr = "address"
    strategies = {"address": [strategy_mock, strategy_mock]}

    mock_df_result = Mock()
    mock_spark = dg._spark_session
    mock_spark.createDataFrame.return_value = mock_df_result

    with patch(
        "dupegrouper.base._process_partition",
        return_value=iter([Row(id="1", address="45th street", email="random@ghs.com", canonical_id=1)]),
    ):
        with patch("dupegrouper.base.WrappedSparkDataFrame") as mockwrapped_df:
            mockwrapped_result = Mock()
            mockwrapped_df.return_value = mockwrapped_result

            dg._canonicalize_spark(attr, strategies)

            # Assertions
            dg._mock_rdd.mapPartitions.assert_called_once()
            mock_spark.createDataFrame.assert_called_once_with(dg._dummy_rdd, schema=ANY)
            mockwrapped_df.assert_called_once_with(mock_df_result, "id")

            assert dg._df == mockwrapped_result


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


@patch("dupegrouper.base.Duped")
def test__process_partition_calls_canonicalize(dupegrouper_mock, partition):
    mock_instance = Mock()
    mock_instance.df = [Row(id=1, canonical_id=0), Row(id=2, canonical_id=0)]
    dupegrouper_mock.return_value = mock_instance

    strategy_mock = Mock()
    strategy_mock.reinstantiate.return_value = strategy_mock
    strategies = {"address": [strategy_mock]}

    result = list(_process_partition(partition, strategies, id="id", attr="address"))

    dupegrouper_mock.assert_called_once()
    mock_instance.apply.assert_called_once_with(strategies)
    mock_instance.canonicalize.assert_called_once_with("address")
    assert result == mock_instance.df


@patch("dupegrouper.base.Duped")
def test__process_partitions_reinstantiated(dupegrouper_mock, partition):
    mock_instance = Mock()
    mock_instance.df = [Row(id=1, canonical_id=0)]
    dupegrouper_mock.return_value = mock_instance

    mock_strategy = Mock()
    mock_strategy.reinstantiate.return_value = "reinstantiated_strategy"

    strategies = {"address": [mock_strategy]}

    result = list(_process_partition(partition, strategies, "id", "address"))

    mock_strategy.reinstantiate.assert_called_once()
    assert result == mock_instance.df


#####################
# TEST canonicalize #
#####################


@pytest.fixture
def dupgrouper_context(request):
    df_type, dfwrapper = request.param
    with patch("dupegrouper.base.wrap"):
        dg = Duped(Mock(spec=df_type), "id")
        dg._df = Mock(spec=dfwrapper)

        with patch.object(dg, "_strategy_manager") as strategy_manager:

            strategy_manager.get.return_value = {"address": [Mock()]}
            strategy_manager.reset.return_value = Mock()
            dg._canonicalize_spark = Mock()
            dg._canonicalize = Mock()

            yield {
                "dg": dg,
                "strategy_manager": strategy_manager,
                "is_spark": "WrappedSparkDataFrame" == dfwrapper.__name__,
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
def test_canonicalize(dupgrouper_context):

    dg = dupgrouper_context["dg"]
    strategy = dupgrouper_context["strategy_manager"]
    is_spark = dupgrouper_context["is_spark"]

    dg.canonicalize("address")

    if is_spark:
        dg._canonicalize_spark.assert_called_once_with("address", strategy.get.return_value)
        dg._canonicalize.assert_not_called()
    else:
        dg._canonicalize.assert_called_once_with("address", strategy.get.return_value)
        dg._canonicalize_spark.assert_not_called()
    strategy.reset.assert_called_once()


##################################
# TEST Duped - public API! #
##################################


def patch_helper_reset(grouper: Duped):
    with patch.object(grouper, "_canonicalize") as mock_canonicalize, patch.object(
        grouper._strategy_manager, "reset"
    ) as mock_reset:

        mock_canonicalize.side_effect = mock_reset

        grouper.canonicalize("address")

        mock_canonicalize.assert_called_once_with("address", ANY)

        grouper._strategy_manager = _StrategyManager()

    assert not grouper.strategies


def test_dupegrouper_strategies_attribute_inline(df_pandas):
    grouper = Duped(df_pandas)

    grouper.apply(Mock(spec=Exact))
    grouper.apply(Mock(spec=Fuzzy))
    grouper.apply((dummy_func, {"str": "random"}))

    assert grouper.strategies == tuple(["Exact", "Fuzzy", "dummy_func"])

    patch_helper_reset(grouper)


def test_dupegrouper_strategies_attribute_dict(df_pandas):
    grouper = Duped(df_pandas)

    grouper.apply(
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
