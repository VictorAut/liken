from unittest.mock import Mock, patch

from pandas.testing import assert_frame_equal
import pytest

from dupegrouper.base import (
    Duped,
    _validate_keep_arg,
    _validate_spark_args,
)


# INITIALIZATION:


@patch("dupegrouper.base.LocalExecutor")
@patch("dupegrouper.base.SparkExecutor")
@patch("dupegrouper.base.wrap")
def test_init_uses_executor(mock_wrap, mock_spark_executor, mock_local_executor, dataframe):
    df, spark = dataframe

    mock_wrap.return_value = Mock()
    dupe = Duped(df, spark_session=spark, keep="first")
    if spark:
        mock_spark_executor.assert_called_once_with(keep="first", spark_session=spark, id=None)
    else:
        mock_local_executor.assert_called_once_with(keep="first")
    mock_wrap.assert_called_once_with(df, None)
    assert dupe._executor is not None


# VALIDATION


@pytest.mark.parametrize("keep", ["first", "last"])
def test_validate_keep_arg_valid(keep):
    assert _validate_keep_arg(keep) == keep


def test_validate_keep_arg_invalid():
    with pytest.raises(ValueError, match="Keep must be one of 'first' or 'last'"):
        _validate_keep_arg("middle")


def test_validate_spark_args_valid(mock_spark_session):
    session = _validate_spark_args(mock_spark_session)
    assert session == mock_spark_session


def test_validate_spark_args_missing_session():
    with pytest.raises(ValueError, match="spark_session must be provided for a spark dataframe"):
        _validate_spark_args(None)


# StrategyManager


@patch("dupegrouper.base.StrategyManager")
def test_apply_delegates_to_strategy_manager(mock, dataframe, strategy_mock):
    df, spark = dataframe

    mock_sm = mock.return_value
    dupe = Duped(df, spark_session=spark)
    dupe.apply({"address": strategy_mock})
    mock_sm.apply.assert_called_once_with({"address": strategy_mock})


# canonicalize


@patch("dupegrouper.base.LocalExecutor")
@patch("dupegrouper.base.wrap")
@patch("dupegrouper.base.StrategyManager")
def test_canonicalize_calls_executor_and_resets_strategy_manager(
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

    dupe = Duped(df, spark_session=spark)
    dupe._executor = mock_executor
    dupe._sm = mock_sm

    dupe.canonicalize(columns="address")

    mock_sm.get.assert_called_once()
    mock_executor.canonicalize.assert_called_once_with(mock_wrap.return_value, "address", {"address": strategy_mock})
    mock_sm.reset.assert_called_once()


# Property attributes


@patch("dupegrouper.base.StrategyManager")
def test_strats_property_returns_manager_output(mock_sm, dataframe):
    df, spark = dataframe
    mock_sm = mock_sm.return_value
    mock_sm.pretty_get.return_value = ("strategy1",)

    dupe = Duped(df, spark_session=spark)
    dupe._sm = mock_sm
    assert dupe.strats == ("strategy1",)


@patch("dupegrouper.base.wrap")
def test_df_property_returns_unwrapped_df(mock_wrap, df_pandas):
    mock_df_wrapper = Mock()
    mock_df_wrapper.unwrap.return_value = df_pandas
    mock_wrap.return_value = mock_df_wrapper

    dupe = Duped(df_pandas)
    assert_frame_equal(dupe.df, df_pandas)
