---
title: Performance
---

Here, **Liken's** scaling with dataset size is explored in detail. Use this as a reference point when building requirements for a deduplication use-case, as different dedupers scale differently.

## Benchmarking

Benchmarked performance is summarised in the below graphic. Dedupers are operated on nominally representative columns of data that might typically be found in a dataset, for example, an "address" column. The below benchmark focuses on performance at the hour-mark as a general maximum. Although runtimes can vary greatly, the hour-long benchmark is a useful one. As such, some deduper's performance are not shown for larger datasets.

Benchmarking was carried out on a standard personal machine.

![Liken](../images/liken-benchmarks.png)

/// caption
Performance of **Liken's** dedupers measured as execution time against increasing dataset sizes. In instances where a dataset size caused a deduper's projected runtime to greatly exceed 1 hour, it was excluded. The single minute and hour marks are provided for orientative benchmarking. The "million-class" dedupers are highlighted in red.
///

## Scaling

Above we same the performance of **Liken's** liken's dedupers. The following graphic provides a normalized view of how the deduper's scale with complexity (dataset size).

![Liken](../images/liken-scaling.png)

/// caption
Computational complexity scaling of Liken's dedupers. 
///

The scaling of deduper's can be useful to provide approximate estimates of the performance of specific deduper's when not provided in the the prior performance graphic. For example, in the case of `cosine` complexity evolves as *O(n^2^)* and it can be estimated that with nominal data, doubling the dataset size from 100K to 200K would results in a four-fold execution time increase i.e. from ~2 hours to ~8 hours.

## Notes

- Individual deduper performance can be greatly affected by the average string length of a column.
- The performance of the `str_*` dedupers is highly performant when selecting patterns that exist sparesely. A generic choice of a pattern such as `'a'` in `str_contains`, for example, results in an explosive runtime increase. Pattern choices should be limited those that have a meaning, for example, `'street'` in an address column. Or, `'ltd'` in a company name column. This is similarily true for `str_len` which is fast when the selected length boundaries do not exist in the data!
- For large datasets, `lsh` is exceptionally performant. Note that for the above benchmarks it is estimated `lsh` could handle 10 million rows in single digit hours, on a standard personal machine.