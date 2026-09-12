"""Build the preprocessed image cache from raw datasets.

Each dataset under data/raw/<name>/ is expected to contain an images/
subdirectory and a labels.csv with columns [filename, grade]. This script
crops each image to its retina FOV, resizes to a fixed size, and writes the
result plus a manifest CSV to data/cache/.

The train/val split assigned here is a local-dev-only stratified split over
whatever sample was downloaded (see split_utils.stratified_split) -- it is
independent of the committed full-dataset splits in data/splits/, which
scripts/make_splits.py produces from the complete raw label set.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

from scripts.split_utils import stratified_split
from src.drscreen.preprocess.retina import preprocess_image

MAX_WORKERS = 4
CACHE_SIZE = 448


def _process_one(task: tuple[str, str, str, int]) -> dict | None:
    dataset, split, filename, grade = task
    raw_path = Path("data/raw") / dataset / "images" / filename
    img = cv2.imread(str(raw_path))
    if img is None:
        print(f"WARNING: unreadable image, skipping: {raw_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        cached_img, _mask = preprocess_image(img, size=CACHE_SIZE)
    except ValueError as e:
        print(f"WARNING: preprocessing failed, skipping {raw_path}: {e}")
        return None

    out_dir = Path("data/cache") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), cv2.cvtColor(cached_img, cv2.COLOR_RGB2BGR))

    return {"path": str(out_path), "dataset": dataset, "split": split, "grade": grade, "has_masks": False}


def build_cache(datasets: list[str], sample: int | None, split: str | None = None) -> pd.DataFrame:
    """`split`: if given (e.g. "test"), every row of every dataset is cached
    under that single split with no train/val stratification -- for a
    held-out test set like Messidor-2, which is never split, only ever
    evaluated whole (see AGENTS.md: test data touched exactly once)."""
    all_rows = []
    for dataset in datasets:
        labels_path = Path("data/raw") / dataset / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"{labels_path} not found. Run the matching scripts/download_*.py first.")
        labels = pd.read_csv(labels_path)
        if sample is not None:
            labels = labels.sample(n=min(sample, len(labels)), random_state=42).reset_index(drop=True)

        if split is not None:
            split_labels = labels.assign(split=split)
        else:
            train_df, val_df = stratified_split(labels, "grade", val_frac=0.15, seed=42)
            train_df = train_df.assign(split="train")
            val_df = val_df.assign(split="val")
            split_labels = pd.concat([train_df, val_df], ignore_index=True)

        tasks = [(dataset, row["split"], row["filename"], row["grade"]) for _, row in split_labels.iterrows()]
        with Pool(processes=min(MAX_WORKERS, len(tasks))) as pool:
            results = pool.map(_process_one, tasks)
        all_rows.extend(r for r in results if r is not None)

    manifest = pd.DataFrame(all_rows)
    manifest_path = Path("data/cache/manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        # Merge with whatever's already cached (e.g. aptos/idrid from an
        # earlier, separate invocation) instead of overwriting it -- this
        # script is called incrementally, once per new dataset, not once
        # for the whole project, so clobbering the existing manifest here
        # would silently destroy rows other configs still depend on.
        existing = pd.read_csv(manifest_path)
        existing = existing[~existing["dataset"].isin(datasets)]
        manifest = pd.concat([existing, manifest], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"cached {len(all_rows)} new images, {len(manifest)} total -> {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["aptos"])
    parser.add_argument("--sample", type=int, default=None, help="cap the number of images per dataset (local dev)")
    parser.add_argument("--split", default=None, help='if given (e.g. "test"), cache every row under this single split, no train/val stratification')
    args = parser.parse_args()
    build_cache(args.datasets, args.sample, args.split)


if __name__ == "__main__":
    main()
