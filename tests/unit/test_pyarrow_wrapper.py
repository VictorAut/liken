"""Unit tests for the pyarrow backend wrapper."""

from __future__ import annotations

import pyarrow as pa

import liken as lk
from liken.backends.pyarrow.backend import PyarrowBackend
from liken.constants import CANONICAL_ID
from liken.core.dispatcher import get_backend


# HELPERS:


def multi_chunk_table() -> pa.Table:
    """A table whose columns are chunked across two record batches."""
    return pa.concat_tables(
        [
            pa.table({"address": ["1 High St", "1 High Street"], "id": ["a", "b"]}),
            pa.table({"address": ["2 Low St"], "id": ["c"]}),
        ]
    )


# DISPATCH:


def test_pyarrow_table_matches_pyarrow_backend():
    assert get_backend(pa.table({"address": ["a"]})).name == "pyarrow"


def test_create_df_infers_types():
    df = PyarrowBackend().create_df([(1, "a", 1.5, True, None)], ["id", "address", "num", "flag", "nul"])

    assert isinstance(df, pa.Table)
    assert df.column_names == ["id", "address", "num", "flag", "nul"]
    assert df.schema.field("id").type == pa.int64()
    assert df.schema.field("address").type == pa.string()
    assert df.schema.field("num").type == pa.float64()
    assert df.schema.field("flag").type == pa.bool_()


def test_create_df_empty_data():
    df = PyarrowBackend().create_df([], ["id", "address"])

    assert isinstance(df, pa.Table)
    assert df.num_rows == 0
    assert df.column_names == ["id", "address"]


# CANONICAL ID:


def _wrap(df: pa.Table, id: str | None = None):
    return PyarrowBackend().wrap(df, id=id)


def test_autoincrement_id():
    wdf = _wrap(pa.table({"address": ["a", "b", "c"]}))

    assert wdf.column(CANONICAL_ID).to_pylist() == [0, 1, 2]
    assert wdf.column(CANONICAL_ID).type == pa.int64()


def test_copy_id_from_id_column():
    df = pa.table({"uid": ["a001", "a002"], "address": ["a", "b"]})

    wdf = _wrap(df, id="uid")

    assert wdf.column(CANONICAL_ID).to_pylist() == ["a001", "a002"]


def test_copy_id_int():
    df = pa.table({"uid": [7, 8], "address": ["a", "b"]})

    wdf = _wrap(df, id="uid")

    assert wdf.column(CANONICAL_ID).to_pylist() == [7, 8]


def test_existing_canonical_id_left_as_is():
    df = pa.table({CANONICAL_ID: ["x", "y"], "address": ["a", "b"]})

    wdf = _wrap(df, id=None)

    assert wdf.column(CANONICAL_ID).to_pylist() == ["x", "y"]


def test_existing_canonical_id_overwritten_by_id_arg():
    df = pa.table({CANONICAL_ID: ["x", "y"], "uid": [10, 20], "address": ["a", "b"]})

    wdf = _wrap(df, id="uid")

    assert wdf.column(CANONICAL_ID).to_pylist() == [10, 20]


# ARROW INTERFACES:


def test_get_col_combines_chunks_into_pa_array():
    wdf = _wrap(multi_chunk_table())

    col = wdf._get_col("address")

    assert isinstance(col, pa.Array)
    assert not isinstance(col, pa.ChunkedArray)
    assert col.to_pylist() == ["1 High St", "1 High Street", "2 Low St"]


def test_get_cols_returns_pa_table():
    wdf = _wrap(multi_chunk_table())

    table = wdf._get_cols(("id", "address"))

    assert isinstance(table, pa.Table)
    assert table.column_names == ["id", "address"]
    assert table.column("id").to_pylist() == ["a", "b", "c"]


# WRAPPER METHODS:


def test_put_col_appends_new_column():
    wdf = _wrap(pa.table({"address": ["a", "b"]}))

    wdf.put_col("score", [1, 2])

    assert wdf.column_names == ["address", CANONICAL_ID, "score"]
    assert wdf.column("score").to_pylist() == [1, 2]


def test_put_col_replaces_existing_column():
    wdf = _wrap(pa.table({"address": ["a", "b"]}))

    wdf.put_col("address", ["z", "y"])

    assert wdf.column("address").to_pylist() == ["z", "y"]


def test_drop_col():
    wdf = _wrap(pa.table({"address": ["a", "b"], "extra": [1, 2]}))

    wdf.drop_col("extra")

    assert wdf.column_names == ["address", CANONICAL_ID]


def test_drop_duplicates_keep_first_preserves_order():
    df = pa.table({CANONICAL_ID: [0, 1, 0, 1, 2], "address": ["a", "b", "c", "d", "e"]})

    wdf = _wrap(df)
    wdf.drop_duplicates("first")

    assert wdf.column("address").to_pylist() == ["a", "b", "e"]


def test_drop_duplicates_keep_last_preserves_original_order():
    df = pa.table({CANONICAL_ID: [0, 1, 0, 1, 2], "address": ["a", "b", "c", "d", "e"]})

    wdf = _wrap(df)
    wdf.drop_duplicates("last")

    assert wdf.column("address").to_pylist() == ["c", "d", "e"]


def test_drop_duplicates_works_on_multi_chunk_table():
    df = pa.concat_tables(
        [
            pa.table({CANONICAL_ID: [1, 2], "address": ["a", "b"]}),
            pa.table({CANONICAL_ID: [1], "address": ["c"]}),
        ]
    )

    wdf = _wrap(df)
    wdf.drop_duplicates("first")

    assert wdf.column("address").to_pylist() == ["a", "b"]


def test_drop_duplicates_null_ids_form_one_group():
    df = pa.table({CANONICAL_ID: [None, None, 1], "address": ["a", "b", "c"]})

    wdf = _wrap(df)
    wdf.drop_duplicates("first")

    assert wdf.column("address").to_pylist() == ["a", "c"]


def test_drop_duplicates_nan_ids_form_one_group():
    df = pa.table({CANONICAL_ID: [float("nan"), float("nan"), 1.0], "address": ["a", "b", "c"]})

    wdf = _wrap(df)
    wdf.drop_duplicates("first")

    assert wdf.column("address").to_pylist() == ["a", "c"]


def test_drop_duplicates_on_empty_table():
    wdf = _wrap(pa.table({"address": []}))
    wdf.drop_duplicates("first")

    assert wdf.num_rows == 0


# SYNTHETIC RECORD:


def test_synthesize_record_first_non_null_per_group():
    df = pa.table(
        {
            CANONICAL_ID: [0, 0, 1],
            "address": ["1 High St", None, "2 Low St"],
            "email": [None, "a@example.com", None],
        }
    )

    wdf = _wrap(df)
    result = wdf.synthesize_record()

    assert isinstance(result, pa.Table)
    assert result.column_names == [CANONICAL_ID, "address", "email"]
    assert result.column(CANONICAL_ID).to_pylist() == [0, 1]
    assert result.column("address").to_pylist() == ["1 High St", "2 Low St"]
    assert result.column("email").to_pylist() == ["a@example.com", None]


def test_synthesize_record_sorted_by_canonical_id():
    df = pa.table(
        {
            CANONICAL_ID: [2, 0],
            "address": ["b", "a"],
        }
    )

    result = _wrap(df).synthesize_record()

    assert result.column(CANONICAL_ID).to_pylist() == [0, 2]
    assert result.column("address").to_pylist() == ["a", "b"]


def test_synthesize_record_works_on_multi_chunk_table():
    df = pa.concat_tables(
        [
            pa.table({CANONICAL_ID: [1, 1], "address": ["a", None]}),
            pa.table({CANONICAL_ID: [2], "address": ["b"]}),
        ]
    )

    result = _wrap(df).synthesize_record()

    assert result.column(CANONICAL_ID).to_pylist() == [1, 2]
    assert result.column("address").to_pylist() == ["a", "b"]


def test_synthesize_record_drops_missing_ids():
    df = pa.table({CANONICAL_ID: [None, float("nan"), 1.0], "address": ["a", "b", "c"]})

    result = _wrap(df).synthesize_record()

    assert result.column(CANONICAL_ID).to_pylist() == [1.0]
    assert result.column("address").to_pylist() == ["c"]


def test_synthesize_record_skips_nan_values():
    df = pa.table(
        {
            CANONICAL_ID: [1.0, 1.0, 2.0],
            "num": [float("nan"), 1.5, 2.5],
        }
    )

    result = _wrap(df).synthesize_record()

    assert result.column("num").to_pylist() == [1.5, 2.5]


def test_synthesize_record_on_empty_table():
    result = _wrap(pa.table({"address": []})).synthesize_record()

    assert result.num_rows == 0
    assert result.column_names == [CANONICAL_ID, "address"]


# END-TO-END:


def test_dedupe_canonicalize_collect_on_pyarrow_input():
    df = pa.table(
        {
            "address": ["1 High St", "1 High St", "2 Low St"],
            "id": ["a", "b", "c"],
        }
    )

    result = lk.dedupe(df).apply(lk.exact()).canonicalize("address", id="id").collect()

    assert isinstance(result, pa.Table)
    assert result.column(CANONICAL_ID).to_pylist() == ["a", "a", "c"]


def test_canonicalize_without_id_uses_uid():
    df = pa.table(
        {
            "address": ["1 High St", "1 High St", "2 Low St"],
            "uid": ["u1", "u2", "u3"],
        }
    )

    result = lk.dedupe(df).apply(lk.exact()).canonicalize("address", id="uid").collect()

    assert result.column(CANONICAL_ID).to_pylist() == ["u1", "u1", "u3"]


def test_canonicals_and_synthesize_on_pyarrow_input():
    df = pa.table(
        {
            "address": ["1 High St", "1 High St", "2 Low St"],
            "uid": ["u1", "u2", "u3"],
        }
    )

    d = lk.dedupe(df).apply(lk.exact()).canonicalize("address", id="uid")

    assert d.canonicals() == {"u1": 2}
    syn = d.synthesize()
    assert isinstance(syn, pa.Table)
    assert syn.column("uid").to_pylist() == ["u1", "u3"]


def test_fake_10_returns_pyarrow_table():
    from liken.datasets import fake_10

    df = fake_10("pyarrow")

    assert isinstance(df, pa.Table)
