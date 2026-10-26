import warnings

import pytest

from liken.dedupers.exact import exact
from liken.exceptions import InvalidArgumentError
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError
from liken.exceptions import InvalidDeduperError
from liken.liken import Dedupe


def test_invalid_argument_type_error_subclasses_builtin_and_base():
    assert issubclass(InvalidArgumentTypeError, TypeError)
    assert issubclass(InvalidArgumentTypeError, InvalidArgumentError)


def test_invalid_argument_value_error_subclasses_builtin_and_base():
    assert issubclass(InvalidArgumentValueError, ValueError)
    assert issubclass(InvalidArgumentValueError, InvalidArgumentError)


def test_invalid_deduper_error_subclasses_builtin_and_base():
    assert issubclass(InvalidDeduperError, TypeError)
    assert issubclass(InvalidDeduperError, InvalidArgumentError)


def test_leaves_catch_via_builtin():
    with pytest.raises(TypeError):
        raise InvalidArgumentTypeError("bad type")
    with pytest.raises(ValueError):
        raise InvalidArgumentValueError("bad value")


def test_leaves_catch_via_base():
    with pytest.raises(InvalidArgumentError):
        raise InvalidArgumentTypeError("bad type")
    with pytest.raises(InvalidArgumentError):
        raise InvalidArgumentValueError("bad value")


def test_invalid_deduper_error_catches_via_base():
    with pytest.raises(InvalidArgumentError):
        raise InvalidDeduperError("bad deduper")


def test_warn_attributes_to_the_users_call_site_via_dedupe(dataframe):
    """`warn` attributes to the user's call through the full public path:
    user code -> `Dedupe.apply` -> `CollectionsManager.apply` -> `warn`."""
    dedupe = Dedupe(dataframe)
    dedupe.apply({"address": exact()})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dedupe.apply({"email": exact()})
    assert len(caught) == 1
    assert caught[0].filename == __file__
