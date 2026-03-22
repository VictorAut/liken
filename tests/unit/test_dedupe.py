from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

from liken._validators import validate_keep_arg
from liken._validators import validate_spark_arg
from liken.liken import Dedupe


# INITIALIZATION:


@patch("liken.liken.LocalExecutor")
@patch("liken.liken.SparkExecutor")
def test_init_uses_executor(mock_spark_executor, mock_local_executor, dataframe):
    df, spark = dataframe

    dupe = Dedupe(df, spark_session=spark)
    if spark:
        mock_spark_executor.assert_called_once_with(spark_session=spark)
    else:
        mock_local_executor.assert_called_once()

    assert dupe._executor is not None


# No apply still exact dedupes


@patch("liken.liken.CollectionsManager")
def test_no_apply_still_exact_apply_once(
    mock_sm,
    dataframe,
):

    df, spark = dataframe

    sm = mock_sm.return_value
    sm.has_applies = False
    sm.is_sequential_applied = True
    sm.get.return_value = {}
    sm.reset.return_value = None

    lk = Dedupe(df, spark_session=spark)
    lk.canonicalize("address")  # <-- no apply!

    sm.apply.assert_called_once()


# validators


@pytest.mark.parametrize("keep", ["first", "last"])
def test_validate_keep_arg_valid(keep):
    assert validate_keep_arg(keep) == keep


def test_validate_keep_arg_invalid():
    with pytest.raises(ValueError, match="Invalid arg: keep must be one of 'first' or 'last'"):
        validate_keep_arg("middle")


def test_validate_spark_arg_valid(mock_spark_session):
    session = validate_spark_arg(mock_spark_session)
    assert session == mock_spark_session


def test_validate_spark_arg_missing_session():
    with pytest.raises(ValueError, match="Invalid arg: spark_session must be provided for a spark dataframe"):
        validate_spark_arg(None)


# Misuse of public API:


def test_validate_df_arg():
    with pytest.raises(ValueError, match="Invalid arg: df must be istance of Pandas, Polars of Spark DataFrames"):
        Dedupe(Mock())  # mock not specced to a df


@patch("liken.liken.wrap")
def test_validate_columns_args_not_used(mock_wrap, deduplication_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="Invalid arg: columns cannot be None"):
        lk = Dedupe(Mock(spec=pd.DataFrame))
        lk.apply(deduplication_mock)
        lk.canonicalize()  # <-- shouldn't be empty


@patch("liken.liken.wrap")
def test_validate_columns_args_not_none(mock_wrap, deduplication_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="Invalid arg: columns must be None"):
        lk = Dedupe(Mock(spec=pd.DataFrame))
        lk.apply({"address": deduplication_mock})  # <-- label here
        lk.canonicalize("address")  # <-- so should not be used here


@patch("liken.liken.wrap")
def test_validate_columns_args_repeated(mock_wrap, deduplication_mock):
    mock_wrap.return_value = Mock()

    with pytest.raises(ValueError, match="Invalid arg: columns labels cannot be repeated"):
        lk = Dedupe(Mock(spec=pd.DataFrame))
        lk.apply(deduplication_mock)
        lk.canonicalize(("email", "email"))  # <-- shouldn't be repeated


# CollectionsManager


@patch("liken.liken.CollectionsManager")
def test_apply_delegates_to_collections_manager(mock, dataframe, deduplication_mock):
    df, spark = dataframe

    mock_sm = mock.return_value
    dupe = Dedupe(df, spark_session=spark)
    dupe.apply({"address": deduplication_mock})
    mock_sm.apply.assert_called_once_with({"address": deduplication_mock})


# canonicalize / drop_duplicates


@patch("liken.liken.LocalExecutor")
@patch("liken.liken.wrap")
@patch("liken.liken.CollectionsManager")
def test_canonicalize_calls(
    mock_sm,
    mock_wrap,
    mock_local,
    dataframe,
    deduplication_mock,
):

    df, spark = dataframe

    mock_wrap.return_value = Mock()
    mock_executor = mock_local.return_value
    mock_sm = mock_sm.return_value
    mock_sm.get.return_value = {"address": deduplication_mock}

    dupe = Dedupe(df, spark_session=spark)
    dupe._executor = mock_executor
    dupe._sm = mock_sm

    dupe.canonicalize("address")

    mock_sm.get.assert_called_once()
    mock_executor.execute.assert_called_once_with(
        mock_wrap.return_value,
        columns="address",
        dedupers={"address": deduplication_mock},
        keep="first",
        drop_duplicates=False,
        drop_canonical_id=False,
        id=None,
    )
    mock_sm.reset.assert_called_once()


@patch("liken.liken.LocalExecutor")
@patch("liken.liken.wrap")
@patch("liken.liken.CollectionsManager")
def test_drop_duplicate_calls(
    mock_sm,
    mock_wrap,
    mock_local,
    dataframe,
    deduplication_mock,
):

    df, spark = dataframe

    mock_wrap.return_value = Mock()
    mock_executor = mock_local.return_value
    mock_sm = mock_sm.return_value
    mock_sm.get.return_value = {"address": deduplication_mock}

    dupe = Dedupe(df, spark_session=spark)
    dupe._executor = mock_executor
    dupe._sm = mock_sm

    dupe.drop_duplicates("address")

    mock_sm.get.assert_called_once()
    mock_executor.execute.assert_called_once_with(
        mock_wrap.return_value,
        columns="address",
        dedupers={"address": deduplication_mock},
        keep="first",
        drop_duplicates=True,
        drop_canonical_id=True,
        id=None,
    )
    mock_sm.reset.assert_called_once()


# Property attributes


@patch("liken.liken.CollectionsManager")
def test_dedupers_property_returns_manager_output(mock_sm, dataframe):
    df, spark = dataframe
    mock_sm = mock_sm.return_value
    mock_sm.pretty_get.return_value = ("deduper1",)

    dupe = Dedupe(df, spark_session=spark)
    dupe._sm = mock_sm
    assert dupe.explain() == ("deduper1",)
