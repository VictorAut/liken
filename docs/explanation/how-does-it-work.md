---
title: Anatomy of Enlace
---

## Pandas and Polars

For single node machines, which is to say when the Pandas or Polars backends are in use, **Enlace** carries the simplified following sequence of events:

``` mermaid
sequenceDiagram
    participant D as Dedupe
    participant SM as StrategyManager
    participant E as LocalExecutor
    participant C as Canonicalizer

    D->>SM: apply()
    D->>SM: apply()
    D->>SM: apply()
    SM->>E: execute(strategies)

    loop For each strategy
        E->>C: run()
        C-->>E: result
    end

    E-->>D: aggregated result
```

## PySpark

For PySpark DataFrames the exact same happens as above, except each the chosen executor is `SparkExecutor` instead of `LocalExecutor`. The Spark executor *re-instantiates* `Dedupe` within each worker node, at which point the `LocalExecutor` takes over again.