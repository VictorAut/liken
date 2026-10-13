import pytest

from liken.exceptions import InvalidArgumentError
from liken.exceptions import InvalidArgumentTypeError
from liken.exceptions import InvalidArgumentValueError
from liken.exceptions import InvalidDeduperError


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
