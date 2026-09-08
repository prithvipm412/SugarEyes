"""Shared stratified train/val split logic used by both the local-dev cache
builder (build_cache.py) and the committed full-dataset splits (make_splits.py)."""
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, label_col: str, val_frac: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(df, test_size=val_frac, stratify=df[label_col], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"grade": rng.integers(0, 5, size=100)})
    train_df, val_df = stratified_split(df, "grade")
    assert len(train_df) + len(val_df) == 100
    assert set(train_df["grade"].unique()) <= set(range(5))
    print("split_utils.py smoke test ok")
