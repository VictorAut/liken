<style>
.md-content .md-typeset h1 { display: none; }
</style>

<p align="center">
  <a href="https://victoraut.github.io/liken/">
    <img src="images/logo-name-dark.png#only-dark" alt="Liken">
    <img src="images/logo-name-light.png#only-light" alt="Liken">
  </a>
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/liken"><img height="20" alt="PyPI Version" src="https://img.shields.io/pypi/v/liken"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/liken">
<img height="20" alt="PyPI Downloads" src="https://static.pepy.tech/badge/liken">
<img height="20" alt="Tests" src="https://img.shields.io/github/actions/workflow/status/VictorAut/liken/ci.yml?label=CI">
<img height="20" alt="Coverage" src="https://img.shields.io/codecov/c/github/VictorAut/liken">
<img height="20" alt="License" src="https://img.shields.io/github/license/VictorAut/liken">
</p>


***
**Source Code**: [https://github.com/VictorAut/liken](https://github.com/VictorAut/liken)
***

<div class="liken-definition">
  <em>Liken</em>:<br>
  <small><i>phrasal verb</i></small><br>
  <small>/ˈlaɪ.kən/</small><br>
  <strong>to say that something is similar to or has the same qualities as something else</strong>
</div>

## Why...

**Liken** provides enhanced deduplication tooling, entity resolution and canonicalization for DataFrames, scaling from single node to massively parallel processed implementations.

### Features

The key features are:

- Near deduplication tooling
- Exploratory duplicate-rate profiling
- Fuzzy string matching deduper
- TF-IDF tokenization deduper
- LSH tokenization deduper
- Jaccard set deduper
- Cosine set deduper
- Pandas API extension
- Composable, rules-based, deduplication pipelines
- Predicate dedupers for rules
- Record linkage and canonicalization
- Built-in Preprocessors
- Pandas, Polars, Modin, Ray, Dask and PySpark support
- Customizable in pure Python
- Synthetic record creation
- Easy to understand syntax
- Dummy datasets for practice

**Liken** aims to answer the call for as-easy-to-use near deduplication as possible, with as natural and easy to understand syntax as possible.

Cut boilerplate code to simple deduplication pipelines with **Liken**.

### Use Cases

- **Find near-duplicate records.** Identify records that represent the same real-world entity despite differences in spelling, formatting or missing values.
- **Perform fuzzy joins between datasets.** Join DataFrames using approximate rather than exact matching to combine imperfect data sources.
- **Link records from multiple data sources.** Connect related records across databases, spreadsheets or data lakes using configurable matching rules.
- **Resolve entities from messy data.** Determine when different records refer to the same person, company, product or other real-world entity.
- **Match names, addresses and companies.** Handle abbreviations, misspellings, inconsistent formatting and other common data quality issues.
- **Canonicalize duplicate records.** Merge groups of duplicate records into a single canonical representation using customizable aggregation rules.
- **Build golden datasets.** Create clean, trusted master datasets by consolidating overlapping information from multiple sources.
- **Clean and standardize messy data.** Apply preprocessing, normalization and deduplication to improve overall data quality.
- **Prepare data for analytics and machine learning.** Reduce duplicate bias and improve data quality before reporting, modelling or downstream processing.

## Supported DataFrame Libraries

<div class="logo-grid">
  <a href="https://pandas.pydata.org" target="_blank">
    <img src="images/supported-libraries/pandas.png" alt="Pandas">
  </a>

  <a href="https://pola.rs" target="_blank">
    <img src="images/supported-libraries/polars.svg" alt="Polars">
  </a>

  <a href="https://modin.readthedocs.io/en/latest/" target="_blank">
    <img src="images/supported-libraries/modin.png" alt="Modin">
  </a>

  <a href="https://spark.apache.org/docs/latest/api/python/" target="_blank">
    <img src="images/supported-libraries/spark.png" alt="PySpark">
  </a>

  <a href="https://docs.ray.io/en/latest/" target="_blank">
    <img src="images/supported-libraries/ray.svg" alt="PySpark">
  </a>

  <a href="https://docs.ray.io/en/latest/" target="_blank">
    <img src="images/supported-libraries/dask.png" alt="Dask">
  </a>

</div>


## Installation

Install with `pip`:

```bash
pip install liken
```

Install with `uv`:

```bash
uv pip install liken
```

### Extras 

**Liken** supports `pandas` and `polars` in the default installation. **Liken** also supports [multiple other DataFrame libraries](./index.md#supported-dataframe-libraries), install them optionally:

=== "pip"

    ```bash
    pip install 'liken[dask]'     # deduplicate dask dataframes
    pip install 'liken[modin]'    # deduplicate modin dataframes
    pip install 'liken[ray]'      # deduplicate ray datasets
    pip install 'liken[pyspark]'  # deduplicate pyspark dataframes
    pip install 'liken[all]'      # deduplicate with any of the above
    ```

=== "uv"

    ```bash
    uv pip install 'liken[dask]'    # deduplicate dask dataframes
    uv pip install 'liken[modin]'   # deduplicate modin dataframes
    uv pip install 'liken[ray]'     # deduplicate ray datasets
    uv pip install 'liken[pyspark]' # deduplicate pyspark dataframes
    uv pip install 'liken[all]'     # deduplicate with any of the above
    ```

??? tip "Installing in a Python project"
    It's recommended you set up a project with [uv](https://docs.astral.sh/uv/getting-started/installation/).

    Install uv, create a project, and install **Liken**:

    ```bash
    uv init my-project
    cd my-project
    uv add liken
    ```

    Use `uv add` to automatically install **Liken** within a new virtual environment, and tracked as a dependency in `pyproject.toml`.

## Use `liken` In Your Code

```python
import liken as lk

# df = ... # e.g. read data

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy())
    .drop_duplicates("name")
)
```
Jump to the [tutorial](tutorials/first-steps.md) to dive deeper into how to build incrementally complex pipelines.

### Pandas Affordances

**Liken's** focus is on composable, complex, deduplication pipelines that scale to distributed datasets. But, extra-easy integration is provided for Pandas DataFrames.

If you are a pandas user looking for intuitive near-deduplication Pandas API extension and little more, head to the [Coming from Pandas?](tutorials/applying-dedupers.md#coming-from-pandas) section!

## AI Agent Skills

**Liken**  makes available agent skills for use in agentic workflows. This is an optional inclusion to your project, and will help you navigate the various APIs so as to best help you solve your problem. 

Install the bundle from the [tessl](https://tessl.io/registry/victoraut/liken-skills) registry:

```bash
tessl install victoraut/liken-skills
```

The bundle contains one skill per API tier:

| Skill | Teaches |
| ----- | ----- |
| `liken` | Overview, and which API to reach for |
| `liken-dedupers` | Applying built-in dedupers |
| `liken-pipelines` | Pipelines with AND/OR/NOT rules and built-in preprocessors |
| `liken-custom-dedupers` | Writing your own dedupers in pure Python |
| `liken-record-linkage` | Canonicalization and synthetic records |
| `liken-backends-performance` | Backend selection, scaling and performance |

??? info "Using the skills"
    Once installed, agent-skill-aware tools (Claude Code, Cursor, and others) discover the skills automatically and load the relevant one on demand. Pin a version for reproducibility, e.g. `tessl install victoraut/liken-skills@0.1.0`. See the [tessl documentation](https://docs.tessl.io) for managing installed skills.

## License

**Liken** is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the [LICENSE](https://github.com/VictorAut/liken/blob/main/LICENSE) file for more details.
