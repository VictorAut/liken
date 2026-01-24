from collections import Counter
from typing import Any, Final, Literal

from pyspark.sql import SparkSession

from enlace._strats_library import BaseStrategy
from enlace._types import Columns


INVALID: Final[str] = "Invalid arg: "
INVALID_SPARK: Final[str] = INVALID + "spark_session must be provided for a spark dataframe"
INVALID_KEEP: Final[str] = INVALID + "keep must be one of 'first' or 'last', got '{}'"
INVALID_STRAT: Final[str] = INVALID + "strat must be instance of BaseStrategy, got {}"
INVALID_COLUMNS_EMPTY: Final[str] = (
    INVALID
    + "columns cannot be None, a column label of tuple of column labels must be provided when using sequential API."
)
INVALID_COLUMNS_REPEATED: Final[str] = (
    INVALID
    + "columns labels cannot be repeated. Repeated labels: '{}'"
)
INVALID_COLUMNS_NOT_NONE: Final[str] = (
    INVALID + "columns must be None when using the dict API, as they have been defined as dictionary keys."
)


# TODO: typing here should be `Any`?
def validate_spark_args(spark_session: SparkSession | None = None, /) -> SparkSession:
    if not spark_session:
        raise ValueError(INVALID_SPARK)
    return spark_session


def validate_keep_arg(keep: Literal["first", "last"]) -> Literal["first", "last"]:

    # TODO: do a type check and TypeError raise here too
    if keep not in ("first", "last"):
        raise ValueError(INVALID_KEEP.format(keep))
    return keep


def validate_strat_arg(strat: Any):
    if not isinstance(strat, BaseStrategy):
        raise TypeError(INVALID_STRAT.format(type(strat).__name__))
    return strat


def validate_columns_arg(
    columns: Columns | None,
    is_sequential_applied: bool,
) -> Columns | None:
    if is_sequential_applied:
        if not columns:
            raise ValueError(INVALID_COLUMNS_EMPTY)

        if isinstance(columns, tuple):
            for label, count in Counter(columns,).items():
                if count > 1:
                    raise ValueError(INVALID_COLUMNS_REPEATED.format(label))

    if not is_sequential_applied and columns:
        raise ValueError(INVALID_COLUMNS_NOT_NONE)
    return columns
