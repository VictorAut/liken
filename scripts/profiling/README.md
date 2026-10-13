# Profilers



## Which tool

| Tool | Script |
| ---- | ------ |
| `cProfile` + `pstats` | `profile.py` |
| `pyinstrument` | `instrument.py` |
| `line_profiler` | `lineprofile.py` |
| profile JSON A/B | `compare.py` |

## Commands

```bash
# cProfile ranking of one run
uv run --active python scripts/profiling/profile.py --deduper fuzzy --size 10000
uv run --active python scripts/profiling/profile.py --deduper exact --size 5000 --csv scrap_data_1M.csv
uv run --active python scripts/profiling/profile.py --deduper tfidf --size 10000 --phase all --dump

# interactive post-hoc analysis of a dump
uv run --active python -m pstats scripts/profiling/results/profile_fuzzy_10k_synthetic_<ts>.pstats

# A/B: profile two checkouts of the same workload, then diff
uv run --active python scripts/profiling/compare.py \
    --input scripts/profiling/results/profile_fuzzy_10k_synthetic_<ts>.json \
            scripts/profiling/results/profile_fuzzy_10k_synthetic_<ts2>.json

# trustworthy shares
uv run --active python scripts/profiling/instrument.py --deduper fuzzy --size 10000

# line detail inside a named hot function (repeatable)
uv run --active python scripts/profiling/lineprofile.py --deduper exact --size 10000 \
    --function liken.liken.Dedupe.canonicalize

# or via the Taskfile
task profile
task profile ARGS="--deduper exact --size 5000"
```


## A/B variations

1. Checkout variant A → `profile.py` → save the JSON.
2. Checkout variant B (same workload flags) → `profile.py` → save the JSON.
3. `compare.py --input <A.json> <B.json>` — the delta table states the two
   git commits it compared.

`compare.py` answers "did the cost move". "Did the run get faster" is a
timing question with noise and repeats — that belongs to the benchmark
scripts (`scripts/benchmarks/`).

## Results

`results/` is gitignored. Files are timestamped and never overwritten:
a second run with the same parameters writes a new file. Large sizes
(1M rows) are slow and produce big JSONs — `--limit` bounds the persisted
rows (`--limit 0` keeps all), and `--csv` covers the heavy-data case.
