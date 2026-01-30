from __future__ import annotations

import os
from typing import Final

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
INVALID_SPARK: Final[str] = INVALID + "spark_session must be provided for a spark dataframe"
INVALID_KEEP: Final[str] = INVALID + "keep must be one of 'first' or 'last', got '{}'"
INVALID_STRAT: Final[str] = INVALID + "strat must be instance of BaseStrategy, got {}"
INVALID_COLUMNS_EMPTY: Final[str] = (
    INVALID
    + "columns cannot be None, a column label of tuple of column labels must be provided when using sequential API."
)
INVALID_COLUMNS_REPEATED: Final[str] = INVALID + "columns labels cannot be repeated. Repeated labels: '{}'"
INVALID_COLUMNS_NOT_NONE: Final[str] = (
    INVALID + "columns must be None when using the dict API, as they have been defined as dictionary keys."
)

# strategy collection errors

INVALID_DICT_KEY_MSG: Final[str] = "Invalid type for dict key type: expected str or tuple, got '{}'"
INVALID_DICT_VALUE_MSG: Final[str] = "Invalid type for dict value: expected list, tuple or 'BaseStrategy', got '{}'"
INVALID_DICT_MEMBER_MSG: Final[str] = (
    "Invalid type for dict value member: at index {} for key '{}': 'expected 'BaseStrategy', got '{}'"
)
INVALID_SEQUENCE_AFTER_DICT_MSG: Final[str] = (
    "Cannot apply a 'BaseStrategy' after a strategy mapping (dict) has been set. "
    "Use either individual 'BaseStrategy' instances or a dict of strategies, not both."
)
INVALID_RULE_EMPTY_MSG: Final[str] = "Rules cannot be empty"
INVALID_RULE_MEMBER_MSG: Final[str] = "Invalid Rules element at index {} is not an instance of On, got '{}'"
INVALID_FALLBACK_MSG: Final[str] = "Invalid strategy: Expected a 'BaseStrategy', a dict or 'Rules', got '{}'"

# strategy collection warnings

WARN_DICT_REPLACES_SEQUENCE_MSG: Final[str] = "Replacing previously added sequence strategy with a dict strategy"
WARN_RULES_REPLACES_RULES_MSG: Final[str] = "Replacing previously added 'Rules' strategy with a new 'Rules' strategy"
