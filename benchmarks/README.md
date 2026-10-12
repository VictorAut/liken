# Benchmarks

Performance scripts for `canonicalize()` timing. Data generation, measurement,
and plotting are separate steps:

- `benchmark.py` times each deduper across a discrete size grid -> JSON.
- `scaling.py` times each deduper over a size sweep -> JSON.
- `plot_benchmark.py` renders grouped bar charts from gathered JSON -> PNG.
- `plot_scaling.py` renders normalised scaling curves from gathered JSON -> PNG.
- `common.py` holds the shared deduper registry, presets, data loading, timed
  runner, and JSON persistence.

## Run

All commands run from the repo root with `uv run --active`.

```bash
# quick exploration
uv run --active python benchmarks/benchmark.py --preset quick
uv run --active python benchmarks/scaling.py --preset quick

# long run, full grid
uv run --active python benchmarks/benchmark.py --preset full

# fine tuning: explicit flags override / replace presets
uv run --active python benchmarks/benchmark.py --csv scrap_data_1_million.csv \
    --datasets 10K 50K --deduper exact tfidf lsh --seed 42
uv run --active python benchmarks/scaling.py --sizes 1000 2000 4000 --deduper exact

# contained runs: another chunk later, same naming convention
uv run --active python benchmarks/benchmark.py --preset full --deduper fuzzy jaccard

# plotting: gather one file, a directory, or a glob; never reruns the benchmark
uv run --active python benchmarks/plot_benchmark.py --input benchmarks/results/benchmark_quick_synthetic_<ts>.json
uv run --active python benchmarks/plot_benchmark.py --input benchmarks/results/
uv run --active python benchmarks/plot_scaling.py --input "benchmarks/results/scaling_*.json" --deduper exact tfidf
```

## Flags

Measurement scripts (`benchmark.py`, `scaling.py`):

| Flag | Meaning |
| ---- | ------- |
| `--preset {quick,full}` | Named preset defining sizes and deduper subset. |
| `--csv <path>` | Load a local CSV instead of synthetic data. |
| `--datasets <sizes...>` | Size grid for `benchmark.py`, e.g. `10K 50K`. Overrides the preset. |
| `--sizes <n...>` | Size sweep for `scaling.py`, e.g. `1000 2000 4000`. Overrides the preset. |
| `--deduper <names...>` | Run only these dedupers. Narrows a preset's subset. |
| `--seed <n>` | Seed for synthetic data. Default is fixed (reproducible runs). |
| `--output <path>` | JSON destination. Default is a timestamped file under `benchmarks/results/`. |

Plot scripts (`plot_benchmark.py`, `plot_scaling.py`):

| Flag | Meaning |
| ---- | ------- |
| `--input <file\|dir\|glob>` | Results files to gather. Required. |
| `--deduper <names...>` | Keep only these dedupers. |
| `--after <timestamp>` | Keep only runs newer than this. |
| `--output <png path>` | PNG destination. Default is timestamped under `benchmarks/results/`. |

Explicit flags take precedence over preset values. Measurement flags may be
combined freely: a run can cover a small chunk (a few dedupers, a few sizes),
stop at any time, and a later run covers another chunk. Plot scripts gather
all matching results and merge them into one chart.

## Data sources

- Default: synthetic data from `liken.datasets` (`fake_row()`-based), extended
  with the registry's extra columns, seeded so runs are reproducible.
- `--csv <path>`: a local CSV, e.g. the kept `scrap_data_1_million.csv`.
  Synthesizing large datasets is slow; use the CSV for big sizes.
- No CSV or data file is versioned. There is no source-data cache.

## Results

Each run writes one JSON file named `{script}_{preset-or-custom}_{source}_{timestamp}.json`
under `benchmarks/results/`. Runs never overwrite an existing results file.
A run is self-contained: its JSON carries the sizes, deduper names, target
columns, timings, seed, and data source needed to plot it. Plot scripts merge
gathered files keyed by (deduper, columns, size); the newest run wins on
overlap (with a warning), and mixed seeds or sources warn too.

Results files and PNGs are gitignored. Never commit script-generated results.

## Presets

| Script | Preset | Sizes | Dedupers |
| ------ | ------ | ----- | -------- |
| benchmark | quick | 10K, 50K | exact, isna, isin, str_contains, str_len |
| benchmark | full | 10K, 50K, 100K, 250K, 1M | full registry |
| scaling | quick | 1000..5000, 6 steps | exact, isna, isin, str_contains, str_len |
| scaling | full | 1000..12000, 12 steps | full registry |

The full registry is defined in `common.py`: `exact`, `fuzzy`, `tfidf`, `lsh`,
`jaccard`, `cosine`, `isna`, `isin`, `str_contains`, `str_startswith`,
`str_endswith`, `str_len`. A deduper entry declares the columns it needs; a
run requesting a deduper whose columns are missing from the data fails with
the list of missing columns.
