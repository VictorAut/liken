from typing import Literal

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
    """Remove leading/trailing whitespace."""
    return Strip()


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
