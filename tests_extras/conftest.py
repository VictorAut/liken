import importlib

import pytest

from liken.core.registries import backends_registry


@pytest.fixture
def expect_backends():

    def _check(expected):
        active = set()

        for name, backend_cls in backends_registry.get_all().items():
            backend = backend_cls()

            try:
                importlib.import_module(backend.name)
                active.add(name)
            except ImportError:
                pass

        assert active == set(expected)

    return _check
