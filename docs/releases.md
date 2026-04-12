---
title: Release History
---
# Release History
## v0.7.1 — 2026-04-12


### Bug Fixes


- docs build and deploy workflow


## v0.7.0 — 2026-04-12


### Features


- added canonicals method

- added synthesize method

- updated to include py3.14

- migrated from mkdocs

- deleted mkdocs.yml

- removed mkdocs dependencies


## v0.6.1 — 2026-04-08


### Bug Fixes


- remove .cache



### Features


- optimised predicate dedupers


## v0.6.0 — 2026-03-24


### Features


- method chaining for On

- refactored fluent

- stubbed processors

- partially created steps

- finished Pipeline class

- renamed strategies to dedupers

- added preprocessors

- wrote tests for preprocessors

- added pipeline unit tests

- partial re-write of docs

- updated pipeline docs

- summarised which API when

- updated api and docs


## v0.5.0 — 2026-03-19


### Features


- updated fuzzy to accept rapidfuzz scorer


## v0.4.5 — 2026-03-18


### Bug Fixes


- broken image links



### Features


- updated badges


## v0.4.4 — 2026-03-18


### Features


- optimised fuzzy

- added dev plotting dependency

- vectorized str match dedupers

- optimised cosine

- added performance explanation

- bumped minor

- linted


## v0.4.3 — 2026-03-06


### Bug Fixes


- removed fuzzy caching


## v0.4.2 — 2026-03-03


### Features


- updated logo


## v0.4.1 — 2026-03-02


### Features


- update logo

- repoint docs link from github to github pages

- dumped patch


## v0.4.0 — 2026-03-01


### Features


- partial implementation of rules predication

- changed docs theme

- Added icon and favicon

- add images

- partially validate rules predication

- partially validated new rules predication

- finished predication

- added how to docs on performance optimisations

- linted

- bumped


## v0.3.1 — 2026-02-22


### Bug Fixes


- uv install dependencies syntax

- locked



### Features


- migrated build backend to uv

- made a lint-slim with no type checking

- bumped to release candidate

- locked uv

- updated to uv publish

- bumped to 0.3.1


## v0.3.0 — 2026-02-22


### Bug Fixes


- merge conflicts


## v0.2.2 — 2026-02-17


### Bug Fixes


- added validation of input dataframe and updated type aliases

- reverted to TypeAlias

- bumped version



### Features


- partially refactored numpy to pyarrow

- pyarrow support for exact deduper on compound columns

- pyarrow backend and removed legacy numpy backend

- bumped version


## v0.2.1 — 2026-02-15


### Bug Fixes


- removed not implemented docstring warnings


## v0.2.0 — 2026-02-15


### Features


- added synthetic datasets


## v0.1.3 — 2026-02-14


### Bug Fixes


- removed action release creation


## v0.1.2 — 2026-02-14


### Bug Fixes


- types in Dedupe class



### Features


- fixed readme and docs typos


## v0.1.1 — 2026-01-31


### Bug Fixes


- broken link in README


## v0.1.0 — 2026-01-30


### Bug Fixes


- drop_duplicates also drops canonical_id column wher strat defined as dict



### Features


- added jaccard

- Added cosine dediper

- refactoring structure to flatten

- flattened structure

- renamed methods

- added tests for LSH and Cosine

- cleaned up tests

- union find testing

- validation mixins for strategies

- refactored base class

- changed abstract method to not implemented method in base strategy class

- added private documentation flags

- split definitions file into types and constants

- renamed strategies file to strats and branched out custom logic into own file

- added public strategy functions

- narrowed types

- Added pair gen custom type

- updated public API

- updated public API apply method

- split strats manager to own file

- partially validated executors module

- fixed return df in LocalExecutor

- functionalised strategy canonical call

- typed Duped constructor with overload

- typed Duped

- added tests

- spark autoincrements canonical id by default

- added TODO comment

- wrote new canonical id rules

- added fake datasets generator

- update dataframe add canonical id method for spark

- removed int() casting of canonical id to allow for str instances

- partial tests for ids

- abstracted canonical id creation to mixin

- typed

- added warnings

- added dataframe tests

- finished ading tests

- added drop duplicated convenience method and delegated keep arg input to .canonicalize

- added mkdocs stubs

- added drop duplicates integration tests

- abstracted union find build and components get into separate functions

- updated test param ids

- boilerplate code for strategy combinations

- implemented str representation of strategies

- implemented and strats

- added tuple strat validation

- validators module

- added deepcopy to StratTuple to avoid state leak

- added tests for On class

- added negation

- removed set_keep injector and passed keep as arg instead

- added Rules tuple wrapper, added tests and refactor error msgs as constants in strats manager

- defined public API packages

- added isna() binary deduper

- refactored data to fake_10 and included null handling

- narrowed integration tests t ofake_10

- added tests for isna() and combinations of isna()

- abstracted get array to _dataframe module instead of strats library

- added string represention of On class and generalised pretty_get to string returns

- changed such that .canonicalize and .drop_duplicates return the dataframe instead of updating in place

- added str_len binary deduper

- updated tests

- no apply still exact dedupes by default

- rename to enlace

- resolved type issues

- wrote docs for ,  and

- added private docstrings to _dataframes and _executors

- linted

- added docstrings for public API exact, fuzzy, tfidf, lsh functions

- removed old pdoc docs

- added public API docstrings for full reference

- added pages

- removed *args pass to Custom

- wrote 'first steps' tutorial

- partially populated deduplication strategies tutotial

- wrote docs for sequential API, dict API and stubbed record linkage

- adjusted pyproject.toml

- finalised rules api docs

- delegated id to canonicalize and added isin strategy

- updated docs for supported backends

- custom strats docs tutorial written

- added docs tutorial on datasets

- added explanation on the different APIs

- finalised explanation

- syntax errors in docs

- finished docs

- renamed to Liken

- updated workflows

- updated version number


## v0.0.0 — 2025-12-31


### Bug Fixes


- redundant regex pattern assignment

- updated poetry lock file



### Features


- added strstartswith deduper

- testing changes

- added ends with str deduped and str contains deduper

- removed print statements

- added unit tests

- upgrade to 0.3.0

- stub cosine distance


<!-- generated by git-cliff -->
