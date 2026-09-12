"""Stratified train/val split of the full APTOS 2019 label set (seed 42),
committed to data/splits/ as reproducibility artifacts.

Messidor-2 gets its own loader (load_messidor2_labels) but is never split or
touched beyond writing its raw labels straight to a test CSV -- it is the
untouched Phase 2 test set (see AGENTS.md: test data is touched exactly once,
at the end).
"""
from pathlib import Path

import pandas as pd

from scripts.split_utils import stratified_split

APTOS_LABELS = Path("data/raw/aptos/labels.csv")
MESSIDOR2_LABELS = Path("data/raw/messidor2/labels.csv")
SPLITS_DIR = Path("data/splits")


def make_aptos_splits() -> None:
    """Regenerates the committed full-APTOS splits from data/raw/aptos/labels.csv.

    Only ever run this against the full ~3662-image Kaggle-side APTOS label
    set (see AGENTS.md: APTOS is trained on Kaggle, never downloaded locally
    in full). A machine with only the small local-dev sample present (e.g.
    Phase 1's 300-image smoke sample) would silently regenerate these
    committed reproducibility artifacts down to that sample -- refuse rather
    than do that silently.
    """
    if not APTOS_LABELS.exists():
        raise FileNotFoundError(f"{APTOS_LABELS} not found. Run the full Kaggle-side APTOS download first.")
    labels = pd.read_csv(APTOS_LABELS)

    existing_train = SPLITS_DIR / "aptos_train.csv"
    existing_val = SPLITS_DIR / "aptos_val.csv"
    if existing_train.exists() and existing_val.exists():
        existing_total = len(pd.read_csv(existing_train)) + len(pd.read_csv(existing_val))
        if len(labels) < existing_total:
            raise RuntimeError(
                f"{APTOS_LABELS} has only {len(labels)} rows, fewer than the "
                f"{existing_total} already committed to {SPLITS_DIR} -- this looks like "
                "the small local-dev sample, not the full Kaggle-side label set. Refusing "
                "to overwrite the committed splits with a smaller one. If this really is "
                "an intentional full-dataset update, delete the existing split files first."
            )

    train_df, val_df = stratified_split(labels, "grade", val_frac=0.15, seed=42)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(existing_train, index=False)
    val_df.to_csv(existing_val, index=False)
    print(f"wrote {len(train_df)} train / {len(val_df)} val rows to {SPLITS_DIR}")


def load_messidor2_labels() -> pd.DataFrame:
    """Load Messidor-2 labels. Test set only -- do not call this during
    Phase 1 or Phase 2 training/threshold selection."""
    if not MESSIDOR2_LABELS.exists():
        raise FileNotFoundError(f"{MESSIDOR2_LABELS} not found. Messidor-2 requires ADCIS registration; see data/splits/README.md.")
    return pd.read_csv(MESSIDOR2_LABELS)


def main() -> None:
    make_aptos_splits()
    if MESSIDOR2_LABELS.exists():
        messidor2 = load_messidor2_labels()
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        messidor2.to_csv(SPLITS_DIR / "messidor2_test.csv", index=False)
        print(f"wrote {len(messidor2)} test rows to {SPLITS_DIR / 'messidor2_test.csv'} (untouched until the single Phase 2 evaluation run)")
    else:
        print(f"{MESSIDOR2_LABELS} not present -- skipping Messidor-2 split (not required until Phase 2; needs ADCIS registration).")


if __name__ == "__main__":
    main()
