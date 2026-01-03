import pandas as pd

from dupegrouper.base import wrap
from dupegrouper.strategies import Custom


# Custom callable function
def my_func(df: pd.DataFrame, attr: str, /, match_str: str) -> dict[str, str]:
    my_map = {}
    for irow, _ in df.iterrows():
        left: str = df.at[irow, attr]
        my_map[left] = left
        for jrow, _ in df.iterrows():
            right: str = df.at[jrow, attr]
            if match_str in left.lower() and match_str in right.lower():
                my_map[left] = right
                break
    return my_map


def test_custom_dedupe(df_pandas):

    deduper = Custom(my_func, "address", match_str="navarra")
    deduper.with_frame(wrap(df_pandas))

    updatedwrapped_df = deduper.dedupe()
    updated_df = updatedwrapped_df.unwrap()

    expected_canonical_ids = [1, 2, 3, 3, 5, 6, 3, 8, 1, 1, 11, 12, 13]

    assert list(updated_df["canonical_id"]) == expected_canonical_ids
