import re
from typing import Final
from typing import Literal
from typing import cast

import nltk
import pyarrow as pa
import pyarrow.compute as pc
from cleanco import basename
from nameparser import HumanName
from nltk.corpus import stopwords as nltk_stopwords
from typing_extensions import override

from liken.exceptions import InvalidArgumentValueError
from liken.validations import require_choice
from liken.validations import require_type


# The languages of nltk's `stopwords` corpus this preprocessor accepts. The
# set mirrors the corpus fileids; see the `remove_stopwords` docstring.
_SUPPORTED_STOPWORD_LANGUAGES: Final[frozenset[str]] = frozenset(
    {
        "albanian",
        "arabic",
        "azerbaijani",
        "basque",
        "belarusian",
        "bengali",
        "catalan",
        "chinese",
        "danish",
        "dutch",
        "english",
        "finnish",
        "french",
        "german",
        "greek",
        "hebrew",
        "hinglish",
        "hungarian",
        "indonesian",
        "italian",
        "kazakh",
        "nepali",
        "norwegian",
        "portuguese",
        "romanian",
        "russian",
        "slovene",
        "spanish",
        "swedish",
        "tajik",
        "tamil",
        "turkish",
        "uzbek",
    }
)

_NORMALIZE_UNICODE_FORMS: Final[tuple[str, ...]] = ("NFC", "NFD", "NFKC", "NFKD")


# BASE:


class Preprocessor:
    """Base class for all preprocessors."""

    def __init__(self) -> None:
        self._init_args: tuple = ()
        self._init_kwargs: dict[str, object] = {}

    def __repr__(self) -> str:
        """Render the preprocessor with its constructor arguments."""
        args = ", ".join(repr(a) for a in self._init_args)
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        joined = ", ".join(filter(None, [args, kwargs]))
        return f"{type(self).__name__}({joined})"

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


class CollapseWhitespace(Preprocessor):
    """Replace every run of whitespace with a single space."""

    @override
    def process(self) -> pa.Array:
        return pc.replace_substring_regex(self._array, r"\s+", " ")


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

    def __init__(
        self,
        form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKD",
    ):
        validated_form = require_type("form", form, str)
        validated_form = require_choice("form", validated_form, _NORMALIZE_UNICODE_FORMS)
        super().__init__()
        self._init_kwargs = {"form": validated_form}
        self._form = cast("Literal['NFC', 'NFKC', 'NFD', 'NFKD']", validated_form)

    @override
    def process(self) -> pa.Array:
        return pc.utf8_normalize(self._array, form=self._form)


class AsciiFold(Preprocessor):
    """Converts alphabetic, numeric, and symbolic characters that are not in
    the Basic Latin Unicode block (first 127 ASCII characters) to their ASCII
    equivalent, if one exists. For example, the filter changes à to a.
    """

    @override
    def process(self) -> pa.Array:
        self._array = pc.utf8_normalize(self._array, form="NFKD")

        return pc.replace_substring_regex(self._array, "[̀-ͯ]", "")


# STOPWORDS PROCESSOR:


class RemoveStopwords(Preprocessor):
    """Remove stopwords

    The `language` argument is validated against a hardcoded set of the
    languages in nltk's `stopwords` corpus. `words` must be a non-empty list
    of `str`; `None` means the language's default stopword list. The nltk
    corpus is downloaded at construction, not lazily at processing.
    """

    def __init__(self, words: list[str] | None = None, language: str = "english"):
        language = require_type("language", language, str)
        language = require_choice("language", language, tuple(sorted(_SUPPORTED_STOPWORD_LANGUAGES)))
        if words is not None:
            require_type("words", words, list)
            if not words:
                raise InvalidArgumentValueError("`words` must be a non-empty list of str, got an empty list")
            for i, word in enumerate(words):
                require_type(f"words[{i}]", word, str)
        # Explicit construction-time download: no lazy download at process.
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
        super().__init__()
        self._init_kwargs = {"language": language} if words is None else {"words": tuple(words), "language": language}
        self._words: tuple[str, ...] | None = tuple(words) if words is not None else None
        self._language = language

    @override
    def process(self) -> pa.Array:
        words = self._words if self._words is not None else tuple(nltk_stopwords.words(self._language))

        pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"

        return pc.replace_substring_regex(self._array, pattern, "")


class RegexReplace(Preprocessor):
    """Replace every match of a regular expression pattern.

    The pattern runs on RE2, not Python's `re`. RE2 guarantees linear-time
    matching, so a pathological pattern fails fast rather than hanging.
    Constructs RE2 does not support, and invalid patterns, raise
    `pa.ArrowInvalid`. The pattern is compiled at construction with Python's
    `re`, which catches invalid syntax eagerly.
    """

    def __init__(self, pattern: str, replacement: str):
        pattern = require_type("pattern", pattern, str)
        replacement = require_type("replacement", replacement, str)
        try:
            re.compile(pattern)
        except re.error as err:
            raise InvalidArgumentValueError(f"`pattern` must be a valid regular expression, got '{pattern}'") from err
        super().__init__()
        self._init_args = (pattern, replacement)
        self._pattern = pattern
        self._replacement = replacement

    @override
    def process(self) -> pa.Array:
        return pc.replace_substring_regex(self._array, self._pattern, self._replacement)


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

        return pa.array([_clean_name(x) if x is not None else None for x in pylist])


class NormalizeCompany(Preprocessor):
    """Normalize company names"""

    @override
    def process(self) -> pa.Array:
        # Important: cast to python list first
        pylist = self._array.to_pylist()

        return pa.array([basename(x) if x is not None else None for x in pylist])


# PUBLIC:


def strip() -> Strip:
    """Remove leading/trailing whitespace."""
    return Strip()


def collapse_whitespace() -> CollapseWhitespace:
    """Replace every run of whitespace with a single space.

    Whitespace runs at the start or end of a string collapse to a single
    space, they are *not* removed. Compose with `strip` for that.

    Whitespace here is ASCII whitespace only: `\\t`, `\\n`, `\\f`, `\\r`
    and space. Other Unicode whitespace, such as non-breaking spaces, is
    not collapsed.
    """
    return CollapseWhitespace()


def lower() -> Lower:
    """Convert strings to lowercase."""
    return Lower()


def alnum() -> Alnum:
    """Remove non-alphanumeric characters, including spaces."""
    return Alnum()


def remove_punctuation() -> RemovePunctuation:
    """Remove punctuation but preserve spaces."""
    return RemovePunctuation()


def normalize_unicode(form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKD") -> NormalizeUnicode:
    """Normalize Unicode strings.

    Args:
        form: Unicode normalization form. Accepted values are "NFC", "NFKC",
            "NFD", "NFKD".
    """
    return NormalizeUnicode(form=form)


def ascii_fold() -> AsciiFold:
    """Converts alphabetic, numeric, and symbolic characters that are not in
    the Basic Latin Unicode block (first 127 ASCII characters) to their ASCII
    equivalent, if one exists. For example, the filter changes à to a.
    """
    return AsciiFold()


def remove_stopwords(
    words: list[str] | None = None,
    language: str = "english",
) -> RemoveStopwords:
    """Remove stopwords.

    Args:
        words: A list of words to ignore. If defined, `language` argument is
            ignored.
        language: The language to use for the stop words dictionary"""
    return RemoveStopwords(words=words, language=language)


def regex_replace(pattern: str, replacement: str) -> RegexReplace:
    """Replace every match of a regular expression pattern.

    Args:
        pattern: The regular expression to match.
        replacement: The string to replace each match with.

    Raises:
        `pa.ArrowInvalid` for an invalid pattern
    """
    return RegexReplace(pattern, replacement)


def normalize_names() -> NormalizeName:
    """Normalize personal names.

    Preserves only first name, middle name and last name. Titles and nicknames
    are stripped. Commas are cleaned.
    """
    return NormalizeName()


def normalize_company() -> NormalizeCompany:
    """Normalize company names.

    Strips common company name nomenclature e.g. "Ltd.", or "LLC".
    """
    return NormalizeCompany()
