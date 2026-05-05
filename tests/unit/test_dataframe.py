from __future__ import annotations

from unittest.mock import Mock

import pytest

from liken.constants import CANONICAL_ID
from liken.core.wrapper import CanonicalIdMixin


# Add Canonical ID


class DummyFrame(CanonicalIdMixin):
    def _df_as_is(self, df): ...
    def _df_overwrite_id(self, df, id): ...
    def _df_copy_id(self, df, id): ...
    def _df_autoincrement_id(self, df): ...
    def _column_labels_list(self, df): ...


PARAMS = [
    (CANONICAL_ID, ["address", CANONICAL_ID], "_df_as_is"),
    ("uid", ["address", CANONICAL_ID], "_df_overwrite_id"),
    (None, ["address", CANONICAL_ID], "_df_as_is"),
    ("uid", ["address", "uid"], "_df_copy_id"),
    (None, ["address"], "_df_autoincrement_id"),
]
IDS = [
    "canonical id already exists; verbose definition",
    "new canonical id as overwrite from other id",
    "canonical id already exists; with warning",
    "new canonical id as write from other id",
    "new autoincremental canonical id",
]


@pytest.mark.parametrize("id, cols, method", PARAMS, ids=IDS)
def test_add_canonical_id_mixin(id, cols, method):

    dummy = DummyFrame()
    dummy._df_as_is = Mock()
    dummy._df_overwrite_id = Mock()
    dummy._df_copy_id = Mock()
    dummy._df_autoincrement_id = Mock()
    dummy._column_labels_list = Mock(return_value=cols)

    df = Mock()

    dummy._add_canonical_id(df, id)

    expected = getattr(dummy, method)
    expected.assert_called_once()
