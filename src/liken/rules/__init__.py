"""The Rules API defines liken's most flexible and powerful approach.

With this API you can construct rules that combine strategies using and
statements.

Additional boolean choice strategies are defined here — they can be powerfully
combined with the `liken` standard deduplication strategies.
"""

from .._collections import Rules
from .._collections import on
from .._dedupers import isin
from .._dedupers import isna
from .._dedupers import str_contains
from .._dedupers import str_endswith
from .._dedupers import str_len
from .._dedupers import str_startswith


__all__ = [
    "Rules",
    "on",
    "isna",
    "isin",
    "str_startswith",
    "str_contains",
    "str_endswith",
    "str_len",
]
