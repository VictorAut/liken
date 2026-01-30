"""This moduel contains argument validation for classes.

Most validations are for public arguments of the 'Dedupe' class.

However, some validations exist for other private classes
"""

from collections import Counter
from typing import Any
from typing import Literal

from pyspark.sql import SparkSession

from liken._constants import INVALID_COLUMNS_EMPTY
from liken._constants import INVALID_COLUMNS_NOT_NONE
from liken._constants import INVALID_COLUMNS_REPEATED
from liken._constants import INVALID_KEEP
from liken._constants import INVALID_SPARK
from liken._constants import INVALID_STRAT
from liken._strats_library import BaseStrategy
from liken._types import Columns


def validate_spark_args(spark_session: SparkSession | None = None, /) -> SparkSession:
    """Validates Spark arg in the 'Dedupe' class"""
    if not spark_session:
        raise ValueError(INVALID_SPARK)
    return spark_session


def validate_keep_arg(keep: Literal["first", "last"]) -> Literal["first", "last"]:
    """Validates Keep arg in the 'Dedupe' class"""
    # TODO: do a type check and TypeError raise here too
    if keep not in ("first", "last"):
        raise ValueError(INVALID_KEEP.format(keep))
    return keep


def validate_strat_arg(strat: Any):
    """Validates that the given 'strategy' is in fact a `BaseStrategy`.

    As used by the strategy manager
    """
    if not isinstance(strat, BaseStrategy):
        raise TypeError(INVALID_STRAT.format(type(strat).__name__))
    return strat


def validate_columns_arg(
    columns: Columns | None,
    is_sequential_applied: bool,
) -> Columns | None:
    """validates inputs to public api 'columns' arg.

    Allowed combinations are:

    - Sequential API: .canonicalize with columns defined
    - Dict API: .canonicalize with no columns defined
    - Rules API: .canonicalize with no columns defined

    Any other combination/repetion raises a value error
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
