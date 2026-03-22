import pyarrow as pa

from liken._preprocessors import Alnum
from liken._preprocessors import Lower
from liken._preprocessors import NormalizeCompany
from liken._preprocessors import NormalizeName
from liken._preprocessors import NormalizeUnicode
from liken._preprocessors import RemovePunctuation
from liken._preprocessors import Stopwords
from liken._preprocessors import Strip


def strip() -> Strip:
    """TODO"""
    return Strip()


def lower() -> Lower:
    """TODO"""
    return Lower()


def alnum() -> Alnum:
    """TODO"""
    return Alnum()


def remove_punctuation() -> RemovePunctuation:
    """TODO"""
    return RemovePunctuation()


def normalize_unicode(form: str = "NFKD") -> NormalizeUnicode:
    """TODO"""
    return NormalizeUnicode(form=form)


def stopwords(words: list[str] | None = None, language: str = "english") -> Stopwords:
    """TODO"""
    return Stopwords(words=words, language=language)


def normalize_names() -> NormalizeName:
    """TODO"""
    return NormalizeName()


def normalize_company() -> NormalizeCompany:
    """TODO"""
    return NormalizeCompany()
