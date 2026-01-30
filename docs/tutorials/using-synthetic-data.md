---
title: Using Synthetic Data
---

## Datasets Sub-Package

**Enlace** provisions dummy datasets for you to play around with. Each dataset must be instantiated with the required [backend](../tutorials/supported-backends.md). Datasets vary by size, starting at 10 rows up to 1 million:

```python
from enlace.datasets import fake_10

df = fake_10(backend="pandas")
```

