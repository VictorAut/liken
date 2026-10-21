"""Liken deduper unit tests"""

from unittest.mock import Mock

import pyarrow as pa
import pytest


@pytest.fixture
def mock_df():
    """
    Minimal LocalDF. Only methods used by dedupers are defined.
    """
    df = Mock()
    df._get_col.return_value = pa.array([1, 2, 3])
    df._get_cols.return_value = pa.array([[1], [2], [3]])
    df.put_col.return_value = df
    df.get_array.return_value = pa.array([1, 2, 3])
    return df
