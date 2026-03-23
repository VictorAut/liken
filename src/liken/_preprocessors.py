import re

import nltk
import pyarrow as pa
import pyarrow.compute as pc
from cleanco import basename
from nameparser import HumanName
from nltk.corpus import stopwords as nltk_stopwords
from typing_extensions import override


# BASE:


class Preprocessor:
    """Base class for all preprocessors."""

    def from_array(self, array: pa.Array) -> None:
        """array setter"""
        self._array = array

    def process(self) -> pa.Array:
        raise NotImplementedError("Child class must implement process()")


# STRING PROCESSORS:


class Strip(Preprocessor):
    """Remove leading/trailing whitespace"""

    @override
    def process(self) -> pa.Array:
        return pc.utf8_trim_whitespace(self._array)


class Lower(Preprocessor):
    """Convert strings to lowercase"""

    @override
    def process(self) -> pa.Array:
        return pc.utf8_lower(self._array)


class Alnum(Preprocessor):
    """Remove non-alphanumeric characters, including spaces"""

    @override
    def process(self) -> pa.Array:
        return pc.replace_substring_regex(self._array, "[^0-9A-Za-z]+", "")


class RemovePunctuation(Preprocessor):
    """Remove punctuation but preserve spaces"""

    @override
    def process(self) -> pa.Array:
        return pc.replace_substring_regex(self._array, r"[^\w\s]+", "")


class NormalizeUnicode(Preprocessor):
    """Normalize Unicode strings"""

    def __init__(self, form: str = "NFKD"):
        self._form = form

    @override
    def process(self) -> pa.Array:
        return pc.utf8_normalize(self._array, form=self._form)


class AsciiFold(Preprocessor):
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


class RemoveStopwords(Preprocessor):
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


class NormalizeName(Preprocessor):
    """Normalize personal names"""

    @override
    def process(self) -> pa.Array:
        def _clean_name(name: str) -> str:

            hn = HumanName(name)
            return f"{hn.first} {hn.middle} {hn.last}".strip()

        # Important: cast to python list first
        pylist = self._array.to_pylist()

        return pa.array(
            [_clean_name(x) if x is not None else None for x in pylist]
        )


class NormalizeCompany(Preprocessor):
    """Normalize company names"""

    @override
    def process(self) -> pa.Array:
        # Important: cast to python list first
        pylist = self._array.to_pylist()

        return pa.array([basename(x) if x is not None else None for x in pylist])
