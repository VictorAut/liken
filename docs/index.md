---
title: Getting Started 
---

Enlace is a library for enhanced deduplication tooling.

The key features are:
- Ability to drop near duplicates from DataFrames
- Ready-to-use deduplication strategies
- Advanced rules based deduplication
- Record linkage and canonicalization
- Pandas, Polars and PySpark support

## Installation

```shell
pip install enlace
```

## Use `enlace` in your code

```python
from enlace import Dedupe

df = pd.DataFrame(columns = ["name"], data = [...])

dp = Dedupe(df)
df = drop_duplicates()
```


## Licence