import re
from typing import Literal

import nltk
import pyarrow as pa
import pyarrow.compute as pc
from cleanco import basename
from nameparser import HumanName
from nltk.corpus import stopwords as nltk_stopwords
from typing_extensions import override


# BASE:


class _Preprocessor:
    """Base class for all preprocessors."""

    def from_array(self, array: pa.Array) -> None:
        """array setter"""
        self._array = array

    def process(self) -> pa.Array:
        raise NotImplementedError("Child class must implement process()")


# STRING PROCESSORS:


class _Strip(_Preprocessor):
    """Remove leading/trailing whitespace"""

    @override
    def process(self) -> pa.Array:
        return pc.utf8_trim_whitespace(self._array)


class _Lower(_Preprocessor):
    """Convert strings to lowercase"""

    @override
    def process(self) -> pa.Array:
        return pc.utf8_lower(self._array)


class _Alnum(_Preprocessor):
    """Remove non-alphanumeric characters, including spaces"""

    @override
    def process(self) -> pa.Array:
        return pc.replace_substring_regex(self._array, "[^0-9A-Za-z]+", "")


class _RemovePunctuation(_Preprocessor):
    """Remove punctuation but preserve spaces"""

    @override
    def process(self) -> pa.Array:
        return pc.replace_substring_regex(self._array, r"[^\w\s]+", "")


class _NormalizeUnicode(_Preprocessor):
    """Normalize Unicode strings"""

    def __init__(
        self,
        form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKD",
    ):
        self._form = form

    @override
    def process(self) -> pa.Array:
        return pc.utf8_normalize(self._array, form=self._form)


class _AsciiFold(_Preprocessor):
    """Converts alphabetic, numeric, and symbolic characters that are not in
    the Basic Latin Unicode block (first 127 ASCII characters) to their ASCII
    equivalent, if one exists. For example, the filter changes à to a.
    """

    def __init__(self, form: str = "NFKD"):
        self._form = form

    @override
    def process(self) -> pa.Array:
        self._array = pc.utf8_normalize(self._array, form="NFKD")

        return pc.replace_substring_regex(self._array, "[̀-ͯ]", "")


# STOPWORDS PROCESSOR:


class _RemoveStopwords(_Preprocessor):
    """Remove stopwords"""

    def __init__(self, words: list[str] | None = None, language: str = "english"):
        self._words = words
        self._language = language

    @override
    def process(self) -> pa.Array:
        if not self._words:
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords")
            self._words = nltk_stopwords.words(self._language)

        pattern = r"\b(" + "|".join(re.escape(w) for w in self._words) + r")\b"

        return pc.replace_substring_regex(self._array, pattern, "")


# NAME PROCESSORS:


class _NormalizeName(_Preprocessor):
    """Normalize personal names"""

    @override
    def process(self) -> pa.Array:
        def _clean_name(name: str) -> str:

            hn = HumanName(name)
            return f"{hn.first} {hn.middle} {hn.last}".strip()

        # Important: cast to python list first
        pylist = self._array.to_pylist()

        return pa.array([_clean_name(x) if x is not None else None for x in pylist])


class _NormalizeCompany(_Preprocessor):
    """Normalize company names"""

    @override
    def process(self) -> pa.Array:
        # Important: cast to python list first
        pylist = self._array.to_pylist()

        return pa.array([basename(x) if x is not None else None for x in pylist])


# PUBLIC:


def strip() -> _Strip:
    """Remove leading/trailing whitespace."""
    return _Strip()


def lower() -> _Lower:
    """Convert strings to lowercase."""
    return _Lower()


def alnum() -> _Alnum:
    """Remove non-alphanumeric characters, including spaces."""
    return _Alnum()


def remove_punctuation() -> _RemovePunctuation:
    """Remove punctuation but preserve spaces."""
    return _RemovePunctuation()


def normalize_unicode(form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKD") -> _NormalizeUnicode:
    """Normalize Unicode strings.

    Args:
        form: Unicode normalization form. Accepted values are "NFC", "NFKC",
            "NFD", "NFKD".
    """
    return _NormalizeUnicode(form=form)


def ascii_fold() -> _AsciiFold:
    """Converts alphabetic, numeric, and symbolic characters that are not in
    the Basic Latin Unicode block (first 127 ASCII characters) to their ASCII
    equivalent, if one exists. For example, the filter changes à to a.
    """
    return _AsciiFold()


def remove_stopwords(
    words: list[str] | None = None,
    language: str = "english",
) -> _RemoveStopwords:
    """Remove stopwords.

    Args:
        words: A list of words to ignore. If defined, `language` argument is
            ignored.
        language: The language to use for the stop words dictionary"""
    return _RemoveStopwords(words=words, language=language)


def normalize_names() -> _NormalizeName:
    """Normalize personal names.

    Preserves only first name, middle name and last name. Titles and nicknames
    are stripped. Commas are cleaned.
    """
    return _NormalizeName()


def normalize_company() -> _NormalizeCompany:
    """Normalize company names.

    Strips common company name nomenclature e.g. "Ltd.", or "LLC".
    """
    return _NormalizeCompany()
