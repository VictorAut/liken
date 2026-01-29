---
title: Enlace
---

**Enlace** is a library providing enhanced deduplication tooling for DataFrames.

The key features are:

- Near deduplication
- Ready-to-use deduplication strategies
- Record linkage and canonicalization
- Rules-based deduplication
- Pandas, Polars and PySpark support
- Customizable in pure Python

## Installation

```shell
pip install enlace
```

## Use `enlace` In Your Code

```python {hl_lines="1 6 7"}
from enlace import Dedupe
import pandas as pd

df = pd.DataFrame(columns = ["name"], data = [...])

dp = Dedupe(df)
df = drop_duplicates()
```


## Licence

**Enlace** is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the LICENSE file for more details.