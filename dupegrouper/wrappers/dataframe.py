"""ABC for wrapped dataframe interfaces"""

from __future__ import annotations
from abc import ABC, abstractmethod
import typing

from dupegrouper.definitions import DataFrameLike, SeriesLike


class WrappedDataFrame(ABC):
    """Container class for a dataframe and associated methods

    At runtime any instance of this class will also be a data container of the
    dataframe. The abstractmethods defined here are all the required
    implementations needed
    """

    def __init__(self, df: DataFrameLike):
        self._df: DataFrameLike = df

    def unwrap(self) -> DataFrameLike:
        return self._df

    @staticmethod
    @abstractmethod
    def _add_group_id(df: DataFrameLike):
        """Return a dataframe with a group id column"""
        pass  # pragma: no cover

    # DATAFRAME `LIBRARY` WRAPPERS:

    @abstractmethod
    def put_col(self, column: str, array) -> typing.Self:
        """assign i.e. write a column with array-like data

        No return; `_df` is updated
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_col(self, column: str) -> SeriesLike:
        """Return a column array-like of data"""
        pass  # pragma: no cover

    @abstractmethod
    def map_dict(self, column: str, mapping: dict) -> SeriesLike:
        """Return a column array-like of data mapped with `mapping`"""
        pass  # pragma: no cover

    @abstractmethod
    def drop_col(self, column: str) -> typing.Self:
        """delete a column with array-like data

        No return: `_df` is updated
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def fill_na(series, array) -> SeriesLike:
        """Return a column array-like of data null-filled with `array`"""
        pass  # pragma: no cover

    # THIN TRANSPARENCY DELEGATION

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._df, name)
