<style>
.md-content .md-typeset h1 { display: none; }
</style>

<p align="center">
  <a href="https://victoraut.github.io/liken/">
    <img src="images/logo-name-dark.png#only-dark" alt="Liken" width="600">
    <img src="images/logo-name-light.png#only-light" alt="Liken" width="600">
  </a>
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/liken"><img height="20" alt="PyPI Version" src="https://img.shields.io/pypi/v/liken"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/liken">
<img height="20" alt="PyPI Downloads per month" src="https://static.pepy.tech/badge/liken/month">
<img height="20" alt="Tests" src="https://img.shields.io/github/actions/workflow/status/VictorAut/liken/ci.yml?label=CI">
<img height="20" alt="Coverage" src="https://img.shields.io/codecov/c/github/VictorAut/liken">
<img height="20" alt="License" src="https://img.shields.io/github/license/VictorAut/liken">
</p>


***
**Source Code**: [https://github.com/VictorAut/liken](https://github.com/VictorAut/liken)
***

## Why...

Liken deduplicates DataFrames, resolves entities and canonicalizes records.

### Features

The key features are:

- Near deduplication tooling
- Exploratory duplicate-rate profiling
- Fuzzy string matching deduper
- TF-IDF tokenization deduper
- LSH tokenization deduper
- Jaccard set deduper
- Cosine set deduper
- Pandas API extension
- Composable, rules-based, deduplication pipelines
- Predicate dedupers for rules
- Record linkage and canonicalization
- Built-in Preprocessors
- Pandas, Polars, Modin, Ray, Dask and PySpark support
- Customizable in pure Python
- Synthetic record creation
- Easy to understand syntax
- Dummy datasets for practice

Liken makes near deduplication as approachable as exact deduplication. Describe the rules, apply them, and get deduplicated or canonicalized DataFrames back.

### Use Cases

- **Find near-duplicate records.** Catch what exact matching misses.
- **Link records that describe the same entity.** Combine datasets and label duplicates with configurable matching rules instead of exact keys.
- **Canonicalize duplicate groups.** Collapse each group of duplicates to a single row and keep a canonical id on every record.
- **Build golden records.** Consolidate each duplicate group into one synthetic, "golden", record.
- **Explore duplicate rates first.** Profile how much duplication each column carries.

## Supported DataFrame Libraries

<div class="logo-grid">
  <a href="https://pandas.pydata.org" target="_blank">
    <img src="images/supported-libraries/pandas.png" alt="Pandas">
  </a>

  <a href="https://pola.rs" target="_blank">
    <img src="images/supported-libraries/polars.svg" alt="Polars">
  </a>

  <a href="https://modin.readthedocs.io/en/latest/" target="_blank">
    <img src="images/supported-libraries/modin.png" alt="Modin">
  </a>

  <a href="https://spark.apache.org/docs/latest/api/python/" target="_blank">
    <img src="images/supported-libraries/spark.png" alt="PySpark">
  </a>

  <a href="https://docs.ray.io/en/latest/" target="_blank">
    <img src="images/supported-libraries/ray.svg" alt="Ray">
  </a>

  <a href="https://docs.dask.org/en/latest/" target="_blank">
    <img src="images/supported-libraries/dask.png" alt="Dask">
  </a>

  <a href="https://arrow.apache.org/docs/python/" target="_blank">
    <img src="images/supported-libraries/arrow.png" alt="Dask">
  </a>

</div>


### Pandas Affordances

Liken's focus is on composable deduplication pipelines that scale to distributed datasets. Pandas users who want intuitive near-deduplication as a pandas API extension get a dedicated entry point: head to the [Coming from Pandas?](tutorials/applying-dedupers.md#coming-from-pandas) section!



## License

Liken is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the [LICENSE](https://github.com/VictorAut/liken/blob/main/LICENSE) file for more details.
