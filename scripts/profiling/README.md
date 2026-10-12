# Profiling liken

Versioned profiling scripts for one liken workload. The workload
(deduper registry, data loading, validation, pipeline build) is imported
from `scripts/benchmarks/common.py` — there is one definition, never a
copy.

## Which tool

| Question | Tool | Script |
| -------- | ---- | ------ |
| Where is the time spent, ranked? (zero installs) | `cProfile` + `pstats` (stdlib) | `profile.py` |
| What are the honest shares? (sampling, low distortion) | `pyinstrument` | `instrument.py` |
| Which line inside one known-hot function? | `line_profiler` | `lineprofile.py` |
| Did the cost move between two versions? | profile JSON A/B | `compare.py` |

`profile.py` is stdlib and always works. `instrument.py` and
`lineprofile.py` need the dev group (`uv sync --all-groups`) and are gated
to `python_full_version < '3.15'` — neither ships cp315 wheels yet.

## Commands

```bash
# first look: cProfile ranking of one run
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

## The method

1. **Fix the workload.** `--deduper` + `--size` + `--seed` + `--source` is
   the unit. A profile only means something against a workload worth
   profiling; "it felt slow" is not a workload.
2. **Profile end to end first.** One `profile.py` or `instrument.py` run of
   the phase under test. Rank the top self-time functions before forming
   any hypothesis. `--phase canonicalize` (default) matches the benchmark
   timing semantics; `--phase all` adds the pipeline build and `apply()`.
3. **Attribute to a named cost centre.** liken's plausible cost centres:
   pair generation (fuzzy scoring, TF-IDF/LSH), preprocessing chains, the
   union-find stage, the canonicaliser's per-row loop, dataframe
   conversion, and backend execution. A bottleneck claim names one of
   these, with a profile share attached.
4. **Verify with the second tool.** cProfile inflates call-heavy code —
   exactly the shape of the per-row loop and the union-find. A hotspot
   claimed from cProfile must survive an `instrument.py` pass before
   optimisation work starts.
5. **Record the profile.** Persisted JSONs carry the workload keys, the git
   commit and dirty flag, and the function rows. Store the summary or the
   artifact in the work item, not just the terminal.

## A/B variations

1. Checkout variant A → `profile.py` → save the JSON.
2. Checkout variant B (same workload flags) → `profile.py` → save the JSON.
3. `compare.py --input <A.json> <B.json>` — the delta table states the two
   git commits it compared.

`compare.py` answers "did the cost move". "Did the run get faster" is a
timing question with noise and repeats — that belongs to the benchmark
scripts (`scripts/benchmarks/`). Both runs must share all workload keys
(deduper, size, seed, source, phase); `compare.py` refuses otherwise.

## Results

`results/` is gitignored. Files are timestamped and never overwritten:
a second run with the same parameters writes a new file. Large sizes
(1M rows) are slow and produce big JSONs — `--limit` bounds the persisted
rows (`--limit 0` keeps all), and `--csv` covers the heavy-data case.

## Out of scope

Memory profilers (memray, tracemalloc), `py-spy`, and the native
distributed dashboards (Spark UI, Ray dashboard, Dask dashboard) are not
installed and not covered here. For distributed backends (dask, modin,
ray, pyspark) the engine's own dashboard is the authoritative view;
these scripts profile pandas.
