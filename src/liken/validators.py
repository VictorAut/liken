"""This moduel contains argument validation for classes.

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
from liken.constants import INVALID_KEEP
from liken.constants import INVALID_PREPROCESSOR
from liken.constants import INVALID_SPARK
from liken.core.deduper import BaseDeduper
from liken.preprocessors import Preprocessor
from liken.types import Columns


if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def validate_spark_arg(spark_session: SparkSession | None = None, /) -> SparkSession:
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


def validate_preprocessor_arg(preprocessor: Preprocessor) -> Preprocessor:
    """Validates that the given arg is in fact a `Preprocessor`"""
    if not isinstance(preprocessor, Preprocessor):
        raise TypeError(INVALID_PREPROCESSOR.format(type(preprocessor).__name__))
    return preprocessor
