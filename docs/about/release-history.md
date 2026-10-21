---
title: Release History
---

## v0.15.0 (2026-10-20)

**Features**

- The `edit_distance` deduper accepts a `metric` option:
  `damerau_levenshtein`, `jaro_winkler`, `indel` or `hamming`, in
  addition to the, previous, default `levenshtein`.
- The `fuzzy` deduper accepts `scorer="jaro_winkler"`.
- New `within` deduper: matches two numbers whose absolute difference is
  at most `tol`.
- New `date_range` deduper: parses values under an explicit `format` and
  matches two dates whose difference is at most `window` days.
- New `sorted_tokens` deduper: matches values built from the same
  whitespace-separated words in any order.


## v0.14.1 (2026-10-20)

**Misc**

- Fixed the `Col` deduper method reference links.


## v0.14.0 (2026-10-19)

**Breaking changes**

- `dedupe()` no longer takes a `spark_session` argument.
- The warning emitted when `spark_session` was passed for a non-spark frame
  no longer exists.

## v0.13.1 (2026-10-19)

**Misc**

- Removed dead code fixes: an unreachable canonical-id overwrite path (and its
  per-backend handlers), an unread wrapper attribute, and a
  no-op exception `__init__` override.


## v0.13.0 (2026-10-19)

**Breaking changes**

- `canonicalize` param `id` is renamed to `source_id`.
- `canonicalize` gains a `canonical_id` argument: the label of the output
  canonical-id column, defaulting to `"canonical_id"`. To reuse a stored
  canonical column, pass it as `source_id`.
- `canonicals()` and `synthesize()` honour the `canonical_id` label stored from
  the last `canonicalize` call.
- `drop_duplicates` never consumes or removes any frame column.
- `explore` no longer mints a default-labelled canonical column on the
  analysed frame. It uses liken's reserved internal label instead.
- The internal label-collision error message is reworded.


## v0.12.3 (2026-10-18)

**Fixes**

- Previously accepted but invalid arguments now raise errors:
    - Empty containers are rejected across all APIs e.g. `apply({})`.
    - `str` is no longer accepted as `isin` deduper values (only a list, tuple, set or
        frozenset).
    - `columns` is type-checked first: only `str` or a tuple of `str` is
        accepted; lists, sets and dicts are rejected with a "use a tuple"
        message, and non-str labels are now type errors.
    - Deduper knobs are validated e.g. `tfidf(topn)` ≥ 1, `lsh(ngram)` ≥ 1,
        and `str_len` `min_len`/`max_len` ≥ 0, amongst others.
        patterns in the three str factories (empty patterns rejected).
    - `fuzzy`deduper `scorer` param is validated against the supported scorers at
        construction.
    - `canonicals(n)` requires an integer ≥ 2; `canonicalize(id)` requires a
        non-empty string; `canonicalize(drop_duplicates=...)` requires a
        bool.
    - The similarity `threshold` range is now the open interval `(0, 1)`.
    - `synthesize` raises a `RuntimeError` when called before `canonicalize`.
    - A preprocessor set on a compound column now raises at execution
        instead of being silently skipped.
    - Preprocessor return values are checked at execution: a preprocessor
        must return a pyarrow Array of the input length, otherwise the error
        names the preprocessor and column.
- Applying a deduper with the reserved `"_default_"` dict key is rejected.
- Missing column labels (`columns`, `id`, dict tuple-key members) raise a
  clear error
- Uniform replace policy for non-accumulating `apply` transitions: every
  transition between the Sequential, Dict and Pipeline APIs now warns and
  replaces.
- Chaining a second deduper on one `Col` warns (last-wins kept).
- A `spark_session` passed with a non-spark dataframe warns.
- Preprocessors record their constructor arguments in `repr`, and `Col`'s
  string representation includes its preprocessors.
- Pipeline steps, `step(preprocessors=...)`, gain explicit semantics: omitted inherits the
  pipeline's preprocessors; `[]` or `None` means no preprocessors for the
  step.
- Validation errors carry liken-meaning exception classes
  collections, preprocessors and the public API.
- Message for a missing Spark session is unified across all raise sites.

## v0.12.2 (2026-10-13)

**Misc**

- Validation errors now carry liken-meaning exception classes.
- Validation messages reworded.
- Values that are `bool` are now rejected wherever a number or integer is
  required (`threshold`, `frac`, `thresholds` items, `max_distance`).

## v0.12.1 (2026-10-13)

**Misc**

- Adds profilers as part of developer tooling


## v0.12.0 (2026-10-12)

**Breaking changes**


- Dataframe backend floors generally changed, and all upper caps are removed
    - pandas floor increased from 1 to 1.5.3.
    - polars floor lowered from 1.24.0 to 0.20.4.
    - pyarrow floor lowered from 23.0.1 to 13.0.0.
    - pyspark floor is unchanged at 4.
    - ray floor lowered from 2.55.0 to 2.50.0.
    - dask floor lowered from 2026.3.0 to 2024.8.0.
    - modin floor lowered from 0.37.1 to 0.22.0.

**Features**


- Python 3.15 support.

## v0.11.0 (2026-10-10)

**Breaking changes**


- Deduper bounding convention is now uniformly inclusive:
    - `str_len` matches lengths in the closed interval `[min_len, max_len]`; a value whose length equals `min_len` now matches (it previously did not), and `min_len == max_len` is the exact-length idiom, matching the new `num_range`'s one-value interval.
    - The threshold family (`fuzzy`, `cosine`, `jaccard`, `tfidf`) matches a pair whose similarity is exactly equal to the threshold (previously strictly above).
- `str_len` validates bounds at construction and raises `ValueError` for malformed calls. Previously `str_len` accepted inverted bounds and silently matched nothing.

**Features**


- New `edit_distance` deduper: matches two values when their Levenshtein distance is at most `max_distance` (default 2).
- New `num_range` deduper: matches numeric values inside the inclusive interval `[min, max]`. Available in it's negated form (`~num_range`).
- Two new preprocessors: `collapse_whitespace` replaces every run of whitespace with a single space; `regex_replace` applies a regular-expression replacement.

**Fixes**


- True null handling: a missing value (`None` or NaN) groups only with another missing value. `"na"` placeholder machinery is removed (so nulls no longer collide with literal `"na"` values)
- Core type contracts annotated: the backend protocol, DataFrame wrapper interfaces and collections.
- `fuzzy`'s `100 * threshold` scaling no longer drifts at the boundary; scores within a 1e-9 tolerance below the boundary also match.


## v0.10.3 (2029-10-08)

**Features**


- Installation guide get's it's own page
- AI Agents documentation relegated to "In Practice"
- General documentation revision


## v0.10.2 (2029-10-07)

**Fixes**


- Typing related imports relegated to checking only
- General Python conventions review


## v0.10.1 (2029-09-30)

**Fixes**


- Missing typing in backends
- Module docstrings, typos fixed
- Validation of keep arg fixed


## v0.10.0 (2029-09-30)

**Features**


- PyArrow Table input support: deduplicate, canonicalize and explore pyarrow Tables natively


## v0.9.2 (2026-09-24)

**Features**


- Documentation overhaul: new documentation pages and added detail on tutorials.
- Additional tests


**Fixes**

- Tooling dependencies pinned
- Pre-commit included


## v0.9.1 (2026-09-23)

**Features**


- Documentation overhaul: positioning now covers entity resolution and canonicalization
- New "Use Cases" section
- Tip on installing Liken in a uv-managed Python project
- Reworked "AI Agent Skills" section, with a fixed tessl registry link
- Docs tooling: `zensical` 0.0.39 to 0.0.51, `rich` added to the docs extras, termynal extension config removed from `zensical.toml`


**Fixes**


- Fixed API reference links in the deduper table in the First Steps tutorial
- CI badge now points at the `ci.yml` workflow


## v0.9.0 (2026-06-04)

**Features**


- Agent skills bundle for agentic workflows, installed from the tessl registry with `tessl install victoraut/liken-skills`
- New `.explore` function for exploratory duplicate-rate profiling
- Skill workspace renamed from `liken/liken-skills` to `victoraut/liken-skills`


**Fixes**


- CI badge in README now points at the `ci.yml` workflow
- Removed the under-development announcement banner from the docs


## v0.8.0 (2026-05-05)

**Features**


- Added support for Modin dataframes
- Added support for Dask dataframes
- Added support for Ray datasets
- Pandas and Polars as default installation, other backends as optional
- Pandas "affordances" added as Pandas API extension
- Documentation on features, installation and first steps updated


## v0.7.4 (2026-04-14)


**Fixes**


- Fix release history


## v0.7.3 (2026-04-14)

**Features**


- Added documentation on "Common Recipes"
- Revised documentation on "Tutorials"


**Fixes**


- Added unit tests for `.synthesize` and `.canonicals`


## v0.7.2 (2026-04-12)

**Features**


- Release history added to documentation About page


## v0.7.1 (2026-04-12)


**Fixes**


- Docs build and deploy workflow


## v0.7.0 (2026-04-12)


**Features**


- `.canonicals` method returns canonical ids with more than one record

- `.synthesize` method returns a "golden" record

- Python 3.14 support

- Migrated from mkdocs to zensical


## v0.6.1 (2026-04-08)


**Fixes**


- Removed .cache dir



**Features**


- Optimised predicate dedupers


## v0.6.0 (2026-03-24)


**Features**


- Pipelines for deduplicating with several steps and AND semantics

- Fluent interfaces

- Pipelines support built-in Preprocessors


## v0.5.0 (2026-03-19)


**Features**


- Fuzzy deduper to accept rapidfuzz scorer


## v0.4.5 (2026-03-18)


**Fixes**


- Broken image links



**Features**


- Updated badges


## v0.4.4 (2026-03-18)


**Features**


- Optimised Fuzzy deduper

- Vectorized str predicate dedupers

- Optimised cosine

- Added performance explanation


## v0.4.3 (2026-03-06)


**Fixes**


- Removed fuzzy caching


## v0.4.2 (2026-03-03)


**Features**


- Updated logo


## v0.4.1 (2026-03-02)


**Features**


- Updated logo


## v0.4.0 (2026-03-01)


**Features**


- Rules predication allows for improved performance when combining a predicate deduper with a similarity deduper

- Added icon and favicon

- Added images


## v0.3.1 (2026-02-22)


**Fixes**


- general uv fixes



**Features**


- Migrated build backend to uv


## v0.3.0 (2026-02-22)


**Fixes**


- Merge conflicts


## v0.2.2 (2026-02-17)


**Fixes**


- Added validation of input dataframe and updated type aliases



**Features**


- Backend migrated from Numpy to Pyarrow


## v0.2.1 (2026-02-15)


**Fixes**


- Removed not implemented docstring warnings


## v0.2.0 (2026-02-15)


**Features**


- Added synthetic datasets


## v0.1.3 (2026-02-14)


**Fixes**


- Removed action release creation


## v0.1.2 (2026-02-14)


**Fixes**


- Types in Dedupe class



**Features**


- Fixed readme and docs typos


## v0.1.1 (2026-01-31)


**Fixes**


- Broken link in README


## v0.1.0 (2026-01-30)


**Fixes**


- drop_duplicates now also drops canonical_id column



**Features**


- New Jaccard deduper

- New Cosine deduper

- New LSH deduper

- Custom deduper implementation

- Added fake datasets generator

- Update dataframe add canonical id method for spark

- Added warnings

- Docs migrated to mkdocs

- Implemented Rules class with and semantics strats

- Added tuple strat validation

- Added negation on On instances

- Added isna binary deduper

- Added str_len binary deduper


## v0.0.0 (2025-12-31)


**Fixes**


- Updated dependencies
