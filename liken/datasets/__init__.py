"""Synthesised data available for experimentation. Makes data available
according to the chosen backend.
"""

from .synthetic import fake_1K
from .synthetic import fake_1M
from .synthetic import fake_10
from .synthetic import fake_100K


__all__ = [
    "fake_10",
    "fake_1K",
    "fake_100K",
    "fake_1M",
]
