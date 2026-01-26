A Python library for grouping duplicate data efficiently.

<p align="center">
<a href="https://pypi.python.org/pypi/dupegrouper"><img height="20" alt="PyPI Version" src="https://img.shields.io/pypi/v/dupegrouper"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dupegrouper">
</p>

# Introduction

TODO

### Ready-to-use deduplication strategies

TODO

### Multiple backends support
Enlace aims to scale in line with your usecase. The following backends are currently support:
- Pandas
- Polars
- PySpark


### A flexible API

Checkout the [API Documentation](https://victorautonell-oiry.me/dupegrouper/dupegrouper.html)


## Installation

```shell
pip install enlace
```

## Example

```python
from enlace import Dedupe, fuzzy

dp = enlace.Dedupe(df)

dp.apply(fuzzy())

df = dp.drop_duplicates("address")
```


# About

## License
This project is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the [LICENSE](LICENSE) file for more details.