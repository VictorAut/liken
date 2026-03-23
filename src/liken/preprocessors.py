from liken._preprocessors import Alnum
from liken._preprocessors import AsciiFold
from liken._preprocessors import Lower
from liken._preprocessors import NormalizeCompany
from liken._preprocessors import NormalizeName
from liken._preprocessors import NormalizeUnicode
from liken._preprocessors import RemovePunctuation
from liken._preprocessors import RemoveStopwords
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


def ascii_fold() -> AsciiFold:
    """TODO"""
    return AsciiFold()


def remove_stopwords(words: list[str] | None = None, language: str = "english") -> RemoveStopwords:
    """TODO"""
    return RemoveStopwords(words=words, language=language)


def normalize_names() -> NormalizeName:
    """TODO"""
    return NormalizeName()


def normalize_company() -> NormalizeCompany:
    """TODO"""
    return NormalizeCompany()
