<p align="center">
<a href="https://pypi.python.org/pypi/liken"><img height="20" alt="PyPI Version" src="https://img.shields.io/pypi/v/liken"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/liken">
</p>

# Introduction

**Liken** is a library providing enhanced deduplication tooling for DataFrames.

The key features are:

- Near deduplication
- Ready-to-use deduplication strategies
- Record linkage and canonicalization
- Rules-based deduplication
- Pandas, Polars and PySpark support
- Customizable in pure Python


## A flexible API

Checkout the [API Documentation](https://victorautonell-oiry.me/liken/liken.html)

## Installation

```shell
pip install liken
```

## Example

```python
from liken import Dedupe, fuzzy

lk = liken.Dedupe(df)

lk.apply(fuzzy())

df = lk.drop_duplicates("address")
```

## License
This project is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the [LICENSE](LICENSE) file for more details.