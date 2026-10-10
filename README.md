<p align="center">
<a href="https://pypi.python.org/pypi/liken"><img height="20" alt="PyPI Version" src="https://img.shields.io/pypi/v/liken"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/liken">
<img height="20" alt="PyPI Downloads per month" src="https://static.pepy.tech/badge/liken/month">
<img height="20" alt="Tests" src="https://img.shields.io/github/actions/workflow/status/VictorAut/liken/ci.yml?label=CI">
<img height="20" alt="Coverage" src="https://img.shields.io/codecov/c/github/VictorAut/liken">
<img height="20" alt="License" src="https://img.shields.io/github/license/VictorAut/liken">
</p>

# Introduction

Liken is a library providing enhanced deduplication tooling for DataFrames.

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


## A flexible API

Checkout the [API Documentation](https://victoraut.github.io/liken/)

## Installation

```shell
uv pip install liken
```

## Example

```python
import liken as lk

df = lk.dedupe(df).apply(lk.fuzzy()).drop_duplicates("address")
```

## License
This project is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the [LICENSE](LICENSE) file for more details.
