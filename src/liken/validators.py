"""This module contains argument validation for classes.

Most validations are for public arguments of the 'Dedupe' class.

However, some validations exist for other private classes
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING
from typing import Literal

from liken.constants import INVALID_COLUMNS_EMPTY
from liken.constants import INVALID_COLUMNS_NOT_NONE
from liken.constants import INVALID_COLUMNS_REPEATED
from liken.constants import INVALID_DEDUPER
from liken.constants import INVALID_EXPLORE_COLUMN_MISSING
from liken.constants import INVALID_EXPLORE_COLUMNS
from liken.constants import INVALID_EXPLORE_DEDUPER
from liken.constants import INVALID_FRAC
from liken.constants import INVALID_KEEP
from liken.constants import INVALID_PREPROCESSOR
from liken.constants import INVALID_SPARK
from liken.constants import INVALID_THRESHOLDS
from liken.core.deduper import BaseDeduper
from liken.core.deduper import CompoundColumnMixin
from liken.core.deduper import ThresholdDeduper
from liken.preprocessors import Preprocessor


if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from liken.types import Columns


def validate_spark_arg(spark_session: SparkSession | None = None, /) -> SparkSession:
    """Validates Spark arg in the 'Dedupe' class"""
    if not spark_session:
        raise ValueError(INVALID_SPARK)
    return spark_session


def validate_keep_arg(keep: Literal["first", "last"]) -> Literal["first", "last"]:
    """Validates Keep arg in the 'Dedupe' class

    Raises:
        TypeError: if `keep` is not a string
        ValueError: if `keep` is a string outside ("first", "last")
    """
    if not isinstance(keep, str):
        raise TypeError(INVALID_KEEP.format(keep))
    if keep not in ("first", "last"):
        raise ValueError(INVALID_KEEP.format(keep))
    return keep


def validate_deduper_arg(deduper: BaseDeduper) -> BaseDeduper:
    """Validates that the given 'deduper' is in fact a `BaseDeduper`.

    As used by the collections manager
    """
    if not isinstance(deduper, BaseDeduper):
        raise TypeError(INVALID_DEDUPER.format(type(deduper).__name__))
    return deduper


def validate_columns_arg(
    columns: Columns | None,
    is_sequential_applied: bool,
) -> Columns | None:
    """validates inputs to public api 'columns' arg.

    Allowed combinations are:

    - Sequential API: .canonicalize with columns defined
    - Dict API: .canonicalize with NO columns defined
    - Pipeline API: .canonicalize with NO columns defined

    Any other combination/repetition raises a value error
    """
    if is_sequential_applied:
        if not columns:
            raise ValueError(INVALID_COLUMNS_EMPTY)

        if isinstance(columns, tuple):
            for label, count in Counter(
                columns,
            ).items():
                if count > 1:
                    raise ValueError(INVALID_COLUMNS_REPEATED.format(label))

    if not is_sequential_applied and columns:
        raise ValueError(INVALID_COLUMNS_NOT_NONE)
    return columns


def validate_preprocessor_arg(preprocessor: Preprocessor) -> Preprocessor:
    """Validates that the given arg is in fact a `Preprocessor`"""
    if not isinstance(preprocessor, Preprocessor):
        raise TypeError(INVALID_PREPROCESSOR.format(type(preprocessor).__name__))
    return preprocessor


def _is_real_number(value: object) -> bool:
    """True for int/float but not bool (which subclasses int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_frac_arg(frac: float) -> float:
    """Validates the `frac` arg of the 'Dedupe.explore' method.

    Must be a number in the range (0, 1].
    """
    if not _is_real_number(frac) or not (0 < frac <= 1):
        raise ValueError(INVALID_FRAC.format(frac))
    return frac


def validate_thresholds_arg(thresholds: list[float]) -> list[float]:
    """Validates the `thresholds` arg of the 'Dedupe.explore' method.

    Must be a non-empty list of numbers, each in the open range (0, 1).
    """
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError(INVALID_THRESHOLDS.format(thresholds))
    if not all(_is_real_number(t) and 0 < t < 1 for t in thresholds):
        raise ValueError(INVALID_THRESHOLDS.format(thresholds))
    return thresholds


def validate_explore_columns_arg(
    columns: list[str] | dict[str, BaseDeduper],
) -> list[str] | dict[str, BaseDeduper]:
    """Validates the `columns` arg of the 'Dedupe.explore' method.

    Accepts either a non-empty list of column labels, or a non-empty dict
    mapping column labels to similarity (threshold) dedupers. Existence of the
    columns in the dataframe is checked separately, at execution time.
    """
    if isinstance(columns, list):
        if not columns or not all(isinstance(c, str) for c in columns):
            raise ValueError(INVALID_EXPLORE_COLUMNS.format(columns))
        return columns

    if isinstance(columns, dict):
        if not columns:
            raise ValueError(INVALID_EXPLORE_COLUMNS.format(columns))
        for label, deduper in columns.items():
            if not isinstance(label, str):
                raise ValueError(INVALID_EXPLORE_COLUMNS.format(columns))
            if not isinstance(deduper, ThresholdDeduper) or isinstance(deduper, CompoundColumnMixin):
                raise ValueError(INVALID_EXPLORE_DEDUPER.format(type(deduper).__name__))
        return columns

    raise ValueError(INVALID_EXPLORE_COLUMNS.format(columns))


def validate_explore_column_exists(column: str, df_columns: list[str]) -> str:
    """Validates that an explored column exists in the dataframe."""
    if column not in df_columns:
        raise ValueError(INVALID_EXPLORE_COLUMN_MISSING.format(column))
    return column
