"""Capture custom deduplication strategies that can be plugged into
`liken.Dedupe`
"""

from .._custom import register


__all__ = ["register"]
