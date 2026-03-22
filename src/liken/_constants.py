from __future__ import annotations

import os
from typing import Final

import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.sql.types import BooleanType
from pyspark.sql.types import DataType
from pyspark.sql.types import DateType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import LongType
from pyspark.sql.types import StringType
from pyspark.sql.types import TimestampType


# CONSTANTS:

# This must be manually maintained in sync with the `UserDataFrame` type
# E.g. if new backend support is added (from a user point of view)
SUPPORTED_DFS = pd.DataFrame | pl.DataFrame | spark.DataFrame

# Default canonical_id label in the dataframe
CANONICAL_ID: Final[str] = os.environ.get("CANONICAL_ID", "canonical_id")

# Placeholder string for Null values
# This is susceptible to erroneous results e.g. 'str_startswith' is used with `pattern`="n"!
NA_PLACEHOLDER: Final[str] = "na"

# Sequential API use will load to this dictionary key by default:
SEQUENTIAL_API_DEFAULT_KEY: Final[str] = "_default_"


# Pyspark sql conversion types
PYSPARK_TYPES: Final[dict[str, DataType]] = {
    "boolean": BooleanType(),
    "date": DateType(),
    "double": DoubleType(),
    "float": FloatType(),
    "int": IntegerType(),
    "bigint": LongType(),
    "string": StringType(),
    "timestamp": TimestampType(),
}

# ERROR MESSAGES

# For argument validations

INVALID: Final[str] = "Invalid arg: "
INVALID_DF: Final[str] = INVALID + "df must be istance of Pandas, Polars of Spark DataFrames, got {}"
INVALID_SPARK: Final[str] = INVALID + "spark_session must be provided for a spark dataframe"
INVALID_KEEP: Final[str] = INVALID + "keep must be one of 'first' or 'last', got {}"
INVALID_DEDUPER: Final[str] = INVALID + "deduper must be instance of BaseDeduper, got {}"
INVALID_PREPROCESSOR: Final[str] = INVALID + "preprocessor must be instance of Preprocessor, got {}"
INVALID_COLUMNS_EMPTY: Final[str] = (
    INVALID
    + "columns cannot be None, a column label of tuple of column labels must be provided when using sequential API."
)
INVALID_COLUMNS_REPEATED: Final[str] = INVALID + "columns labels cannot be repeated. Repeated labels: '{}'"
INVALID_COLUMNS_NOT_NONE: Final[str] = (
    INVALID + "columns must be None when using the dict API, as they have been defined as dictionary keys."
)

# collection errors

INVALID_DICT_KEY_MSG: Final[str] = "Invalid type for dict key type: expected str or tuple, got '{}'"
INVALID_DICT_VALUE_MSG: Final[str] = "Invalid type for dict value: expected list, tuple or 'BaseDeduper', got '{}'"
INVALID_DICT_MEMBER_MSG: Final[str] = (
    "Invalid type for dict value member: at index {} for key '{}': 'expected 'BaseDeduper', got '{}'"
)
INVALID_SEQUENCE_AFTER_DICT_MSG: Final[str] = (
    "Cannot apply a 'BaseDeduper' after a deduper mapping (dict) has been set. "
    "Use either individual 'BaseDeduper' instances or a dict of dedupers, not both."
)
INVALID_RULE_EMPTY_MSG: Final[str] = "Pipeline cannot be empty"
INVALID_RULE_MEMBER_MSG: Final[str] = "Invalid Pipeline element at index {} is not an instance of On, got '{}'"
INVALID_FALLBACK_MSG: Final[str] = "Invalid deduper: Expected a 'BaseDeduper', a dict or 'Pipeline', got '{}'"

# collection warnings

WARN_DICT_REPLACES_SEQUENCE_MSG: Final[str] = "Replacing previously added sequence deduper with a dict deduper"
WARN_RULES_REPLACES_RULES_MSG: Final[str] = (
    "Replacing previously added 'Pipeline' deduper with a new 'Pipeline' deduper"
)
