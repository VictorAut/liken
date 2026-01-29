---
title: Enlace
---

**Enlace** is a library providing enhanced deduplication tooling for DataFrames

The key features are:

- Near deduplication :material-check:
- Ready-to-use deduplication strategies :material-check:
- Record linkage and canonicalization :material-check:
- Rules-based deduplication :material-check:
- Pandas, Polars and PySpark support :material-check:
- Customizable in pure Python :material-check:

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