---
title: Getting Started 
---

Enlace is a library providing enhanced deduplication tooling for DataFrames

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

```python {hl_lines="1 6 7"}
from enlace import Dedupe
import pandas as pd

df = pd.DataFrame(columns = ["name"], data = [...])

dp = Dedupe(df)
df = drop_duplicates()
```


## Licence