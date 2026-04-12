---
title: Release History
---

## v0.7.2 (2026-04-12)

**Features**


- Release history added to documentation About page


## v0.7.1 (2026-04-12)


**Fixes**


- Docs build and deploy workflow


## v0.7.0 (2026-04-12)


**Features**


- `.canonicals` method returns canonical ids with more than one record

- `.synthesize method returns a "golden" record

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