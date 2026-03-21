import pyarrow as pa

from liken._processors import Alnum
from liken._processors import Lower
from liken._processors import NormalizeCompany
from liken._processors import NormalizeName
from liken._processors import NormalizeUnicode
from liken._processors import RemovePunctuation
from liken._processors import Stopwords
from liken._processors import Strip


def strip() -> pa.Array:
    """TODO"""
    return Strip()


def lower() -> pa.Array:
    """TODO"""
    return Lower()


def alnum() -> pa.Array:
    """TODO"""
    return Alnum()


def remove_punctuation() -> pa.Array:
    """TODO"""
    return RemovePunctuation()


def normalize_unicode(form: str = "NFKD") -> pa.Array:
    """TODO"""
    return NormalizeUnicode(form=form)


def stopwords(words: list[str] | None = None, language: str = "english") -> pa.Array:
    """TODO"""
    return Stopwords(words=words, language=language)


def normalize_names() -> pa.Array:
    """TODO"""
    return NormalizeName()


def normalize_company() -> pa.Array:
    """TODO"""
    return NormalizeCompany()
