from unittest.mock import Mock, patch

import pytest
from pandas.testing import assert_frame_equal

from enlace._validators import validate_keep_arg, validate_spark_args
from enlace.dedupe import Dedupe


# INITIALIZATION:


@patch("enlace.dedupe.LocalExecutor")
@patch("enlace.dedupe.SparkExecutor")
@patch("enlace.dedupe.wrap")
def test_init_uses_executor(mock_wrap, mock_spark_executor, mock_local_executor, dataframe):
    df, spark = dataframe

    mock_wrap.return_value = Mock()
    dupe = Dedupe(df, spark_session=spark)
    if spark:
        mock_spark_executor.assert_called_once_with(spark_session=spark, id=None)
    else:
        mock_local_executor.assert_called_once()
    mock_wrap.assert_called_once_with(df, None)
    assert dupe._executor is not None


# No apply still exact dedupes


@patch("enlace.dedupe.StrategyManager")
def test_no_apply_still_exact_apply_once(
    mock_sm,
    dataframe,
):
    
    df, spark = dataframe
    
    sm = mock_sm.return_value
    sm._has_applies = False
    sm.is_sequential_applied = True
    sm.get.return_value = {}
    sm.reset.return_value = None

    dp = Dedupe(df, spark_session=spark)
    dp.canonicalize("address") # <-- no apply!

    sm.apply.assert_called_once() 

# validators


@pytest.mark.parametrize("keep", ["first", "last"])
def test_validate_keep_arg_valid(keep):
    assert validate_keep_arg(keep) == keep


def test_validate_keep_arg_invalid():
    with pytest.raises(ValueError, match="Invalid arg: keep must be one of 'first' or 'last'"):
        validate_keep_arg("middle")


def test_validate_spark_args_valid(mock_spark_session):
    session = validate_spark_args(mock_spark_session)
    assert session == mock_spark_session


def test_validate_spark_args_missing_session():
    with pytest.raises(ValueError, match="Invalid arg: spark_session must be provided for a spark dataframe"):
        validate_spark_args(None)


# Misuse of API:


@patch("enlace.dedupe.wrap")
def test_validate_columns_args_not_used(mock_wrap, strategy_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="Invalid arg: columns cannot be None"):
        dp = Dedupe(Mock())
        dp.apply(strategy_mock)
        dp.canonicalize()  # <-- shouldn't be empty


@patch("enlace.dedupe.wrap")
def test_validate_columns_args_not_none(mock_wrap, strategy_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="Invalid arg: columns must be None"):
        dp = Dedupe(Mock())
        dp.apply({"address": strategy_mock})  # <-- label here
        dp.canonicalize("address")  # <-- so should not be used here


@patch("enlace.dedupe.wrap")
def test_validate_columns_args_repeated(mock_wrap, strategy_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="Invalid arg: columns labels cannot be repeated"):
        dp = Dedupe(Mock())
        dp.apply(strategy_mock)
        dp.canonicalize(("email", "email"))  # <-- shouldn't be repeated


# StrategyManager


@patch("enlace.dedupe.StrategyManager")
def test_apply_delegates_to_strategy_manager(mock, dataframe, strategy_mock):
    df, spark = dataframe

    mock_sm = mock.return_value
    dupe = Dedupe(df, spark_session=spark)
    dupe.apply({"address": strategy_mock})
    mock_sm.apply.assert_called_once_with({"address": strategy_mock})


# canonicalize / drop_duplicates


@patch("enlace.dedupe.LocalExecutor")
@patch("enlace.dedupe.wrap")
@patch("enlace.dedupe.StrategyManager")
def test_canonicalize_calls(
    mock_sm,
    mock_wrap,
    mock_local,
    dataframe,
    strategy_mock,
):

    df, spark = dataframe

    mock_wrap.return_value = Mock()
    mock_executor = mock_local.return_value
    mock_sm = mock_sm.return_value
    mock_sm.get.return_value = {"address": strategy_mock}

    dupe = Dedupe(df, spark_session=spark)
    dupe._executor = mock_executor
    dupe._sm = mock_sm

    dupe.canonicalize("address")

    mock_sm.get.assert_called_once()
    mock_executor.execute.assert_called_once_with(
        mock_wrap.return_value,
        columns="address",
        strats={"address": strategy_mock},
        keep="first",
        drop_duplicates=False,
        drop_canonical_id=False,
    )
    mock_sm.reset.assert_called_once()


@patch("enlace.dedupe.LocalExecutor")
@patch("enlace.dedupe.wrap")
@patch("enlace.dedupe.StrategyManager")
def test_drop_duplicate_calls(
    mock_sm,
    mock_wrap,
    mock_local,
    dataframe,
    strategy_mock,
):

    df, spark = dataframe

    mock_wrap.return_value = Mock()
    mock_executor = mock_local.return_value
    mock_sm = mock_sm.return_value
    mock_sm.get.return_value = {"address": strategy_mock}

    dupe = Dedupe(df, spark_session=spark)
    dupe._executor = mock_executor
    dupe._sm = mock_sm

    dupe.drop_duplicates("address")

    mock_sm.get.assert_called_once()
    mock_executor.execute.assert_called_once_with(
        mock_wrap.return_value,
        columns="address",
        strats={"address": strategy_mock},
        keep="first",
        drop_duplicates=True,
        drop_canonical_id=True,
    )
    mock_sm.reset.assert_called_once()


# Property attributes


@patch("enlace.dedupe.StrategyManager")
def test_strats_property_returns_manager_output(mock_sm, dataframe):
    df, spark = dataframe
    mock_sm = mock_sm.return_value
    mock_sm.pretty_get.return_value = ("strategy1",)

    dupe = Dedupe(df, spark_session=spark)
    dupe._sm = mock_sm
    assert dupe.strats == ("strategy1",)


@patch("enlace.dedupe.wrap")
def test_df_property_returns_unwrapped_df(mock_wrap, df_pandas):
    mock_df_wrapper = Mock()
    mock_df_wrapper.unwrap.return_value = df_pandas
    mock_wrap.return_value = mock_df_wrapper

    dupe = Dedupe(df_pandas)
    assert_frame_equal(dupe.df, df_pandas)
