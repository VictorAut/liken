from __future__ import annotations

from typing import Final


# Default canonical_id label in the dataframe
CANONICAL_ID: Final[str] = "canonical_id"

# Sequential API use will load to this dictionary key by default:
SEQUENTIAL_API_DEFAULT_KEY: Final[str] = "_default_"


# ERROR MESSAGES

# fmt: off

# spark session

SPARK_SESSION_REQUIRED_MSG: Final[str] = "spark_session arg must be provided for a spark dataframe"
WARN_SPARK_SESSION_IGNORED_MSG: Final[str] = "spark_session provided for a non-spark dataframe; the argument was ignored"

# collection errors
INVALID_COLUMNS_REQUIRED_MSG: Final[str] = "columns cannot be None; a column label or tuple of column labels must be provided when using the sequential API"
INVALID_COLUMNS_NOT_NONE_MSG: Final[str] = "columns must be None when using the dict API, as they have been defined as dictionary keys"
INVALID_DICT_KEY_MSG: Final[str] = "Invalid type for dict key type: expected str or tuple, got '{}'"
INVALID_DICT_VALUE_MSG: Final[str] = "Invalid type for dict value: expected a deduper (list, tuple or 'BaseDeduper'), got '{}'"
INVALID_DICT_MEMBER_MSG: Final[str] = "Invalid type for dict value member: at index {} for key '{}': expected 'BaseDeduper', got '{}'"
INVALID_SEQUENCE_AFTER_DICT_MSG: Final[str] = "Cannot apply a 'BaseDeduper' after a deduper mapping (dict) has been set. Use either an individual 'BaseDeduper' instances or a dict of dedupers, not both."
INVALID_APPLY_AFTER_PIPELINE_MSG: Final[str] = "Cannot apply a 'BaseDeduper' after a 'Pipeline' deduper has been set. Use either individual 'BaseDeduper' instances or a 'Pipeline' deduper, not both."
INVALID_FALLBACK_MSG: Final[str] = "Invalid deduper: Expected a 'BaseDeduper', a dict or 'Pipeline', got '{}'"
INVALID_EMPTY_DICT_MSG: Final[str] = "Invalid deduper dict: expected a non-empty dict of column labels to dedupers, got an empty dict"
INVALID_EMPTY_DICT_VALUE_MSG: Final[str] = "Invalid deduper dict value: key '{}' maps to an empty sequence; expected at least one deduper"
INVALID_EMPTY_PIPELINE_MSG: Final[str] = "Invalid pipeline: expected at least one step, got an empty pipeline"
INVALID_EMPTY_PIPELINE_STEP_MSG: Final[str] = "`cols` must be a non-empty list, got an empty list"
INVALID_RESERVED_DICT_KEY_MSG: Final[str] = "Invalid dict key: '{}' is reserved for the sequential API"
INVALID_NO_DEDUPER_CHAINED_MSG: Final[str] = "Invalid 'Col': no deduper chained yet; chain a deduper method first"

# collection warnings

WARN_DICT_REPLACES_SEQUENCE_MSG: Final[str] = "Replacing previously added sequence deduper with a dict deduper"
WARN_DICT_REPLACES_DICT_MSG: Final[str] = "Replacing previously added dict deduper with a new dict deduper"
WARN_DICT_REPLACES_PIPELINE_MSG: Final[str] = "Replacing previously added 'Pipeline' deduper with a dict deduper"
WARN_PIPELINE_REPLACES_SEQUENCE_MSG: Final[str] = "Replacing previously added sequence deduper with a 'Pipeline' deduper"
WARN_PIPELINE_REPLACES_DICT_MSG: Final[str] = "Replacing previously added dict deduper with a 'Pipeline' deduper"
WARN_RULES_REPLACES_RULES_MSG: Final[str] = "Replacing previously added 'Pipeline' deduper with a new 'Pipeline' deduper"
WARN_COL_RECHAINED_MSG: Final[str] = "Chaining a second deduper on the same 'Col' replaces the previously chained deduper; only the last one applies"
WARN_COL_REUSED_MSG: Final[str] = "The same 'Col' is used in more than one step; each step re-applies its deduper"

# fmt: on
