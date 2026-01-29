---
title: Using Synthetic Data
---

## Datasets Sub-package

**Enlace** provisions dummy datasets for you to play around with. Each dataset must instantiated with the required [backend](../concepts/supported-backends.md). Datasets vary by size, starting at 10 rows up to 1 million:

```python
from enlace.datasets import fake_10

df = fake_10(backend="pandas")
```

