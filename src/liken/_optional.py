import importlib
from types import ModuleType


def import_optional_dependency(
    name: str,
) -> ModuleType | None:

    msg = f"`Import {name}` failed."
    try:
        module = importlib.import_module(name)
    except ImportError as err:
        raise ImportError(msg) from err

    return module
