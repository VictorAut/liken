"""Defines collections of dedupers"""

from __future__ import annotations

from collections import UserDict
from typing import final

from liken.constants import INVALID_DICT_KEY_MSG
from liken.constants import INVALID_DICT_MEMBER_MSG
from liken.constants import INVALID_DICT_VALUE_MSG
from liken.core.deduper import BaseDeduper
from liken.exceptions import InvalidDeduperError


# DICT CONFIG:


@final
class DeduplicationDict(UserDict):
    """Dict collection for dedupers in the Sequential and Dict APIs

    For Sequential API all values (dedupers) are added under a default key.

    For Dict API column label(s) (i.e. str or tuple) are the keys."""

    def __setitem__(self, key, value):
        if not isinstance(key, str | tuple):
            raise InvalidDeduperError(INVALID_DICT_KEY_MSG.format(type(key).__name__))
        if not isinstance(value, list | tuple | BaseDeduper):
            raise InvalidDeduperError(INVALID_DICT_VALUE_MSG.format(type(value).__name__))
        if not isinstance(value, BaseDeduper):
            for i, member in enumerate(value):
                if not isinstance(member, BaseDeduper):
                    raise InvalidDeduperError(INVALID_DICT_MEMBER_MSG.format(i, key, type(member).__name__))
        else:
            value = (value,)
        super().__setitem__(key, value)

    def __str__(self):
        rep = ""
        for k, values in self.items():
            krep = ""
            for v in values:
                krep += "\n\t\t" + str(v) + ","
            rep += f"\n\t'{k}': ({krep[:-1]},\n\t\t),"
        return "{" + rep + "\n}"
