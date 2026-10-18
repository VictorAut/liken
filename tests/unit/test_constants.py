import inspect

import liken.constants
from liken.constants import CANONICAL_ID


def test_canonical_id_is_the_default_label():
    assert CANONICAL_ID == "canonical_id"


def test_canonical_id_is_not_read_from_the_environment(monkeypatch):
    """The label is a module constant, not an env override."""
    monkeypatch.setenv("CANONICAL_ID", "random_id")

    assert liken.constants.CANONICAL_ID == "canonical_id"


def test_constants_module_does_not_read_the_environment():
    source = inspect.getsource(liken.constants)

    assert "os." not in source
