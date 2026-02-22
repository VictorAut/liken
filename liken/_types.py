"""Shared liken types"""

from __future__ import annotations

from typing import Literal
from typing import TypeAlias


Columns: TypeAlias = str | tuple[str, ...]  # label(s) that identify attributes of a dataframe for deduplication
Keep: TypeAlias = Literal["first", "last"]  # Canonicalisation rule
SimilarPairIndices: TypeAlias = tuple[int, int]
