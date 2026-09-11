"""Download a small local development sample of APTOS 2019.

Full APTOS 2019 training only ever happens on Kaggle notebooks (see
AGENTS.md) -- this script pulls just enough images locally to develop and
smoke-test the Phase 1 pipeline on this machine.

Requires Kaggle API credentials: either ~/.kaggle/kaggle.json or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables.
See https://www.kaggle.com/docs/api for how to generate a key.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/aptos")
DEFAULT_DATASET = "mariaherrerot/aptos2019"


def _find_label_csv(download_dir: Path) -> Path:
    """Pick the training label CSV specifically -- this dataset mirror ships
    train_1.csv / valid.csv / test.csv side by side, so "starts with train"
    isn't enough; we also have to exclude anything that merely contains
    "train" as a substring of something else. Preferring the shortest
    train-flagged match keeps this robust across dataset mirrors."""
    candidates = sorted(download_dir.rglob("*.csv"))
    train_candidates = [c for c in candidates if "train" in c.stem.lower() and "test" not in c.stem.lower()]
    if train_candidates:
        return min(train_candidates, key=lambda p: len(p.stem))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No label CSV found under {download_dir} after download")


def _find_image_dir(download_dir: Path, keyword: str) -> Path:
    """Pick the image directory matching `keyword` (e.g. "train"), preferring
    the deepest match -- datasets often nest e.g. train_images/train_images/
    with the outer folder just a wrapper around the real image directory."""
    candidates = [p for p in download_dir.rglob("*") if p.is_dir() and "image" in p.name.lower()]
    keyed = [c for c in candidates if keyword in c.as_posix().lower()]
    pool = keyed or candidates
    if not pool:
        raise FileNotFoundError(f"No {keyword} image directory found under {download_dir}")
    return max(pool, key=lambda p: len(p.parts))


def _all_label_rows(download_dir: Path) -> pd.DataFrame:
    """All labeled rows across every label CSV found, id-deduplicated.
    Mirrors like this one split labels across train_1.csv/valid.csv/test.csv
    -- a lookup for a specific, predetermined set of ids (rather than "N
    random training images") needs to search all of them, not just the
    train-flagged file."""
    frames = []
    for csv_path in sorted(download_dir.rglob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            continue
        id_col = "id_code" if "id_code" in df.columns else df.columns[0]
        grade_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]
        frames.append(df[[id_col, grade_col]].rename(columns={id_col: "id", grade_col: "grade"}))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="id")


def _find_image_anywhere(download_dir: Path, image_id: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        matches = list(download_dir.rglob(f"{image_id}{ext}"))
        if matches:
            return matches[0]
    return None


def download_sample(n_samples: int, dataset_slug: str, out_dir: Path = RAW_DIR, ids: set[str] | None = None) -> None:
    """Download images into `out_dir`. If `ids` is given, download exactly
    those (bare, extension-free) ids instead of a random `n_samples` sample
    -- used to pull a specific, bounded set (e.g. the real Kaggle-trained
    model's 551-image validation split) rather than the whole dataset. The
    ids path searches every label CSV and image directory in the download
    (not just the "train" ones), since a predetermined id could be in any
    of a mirror's train/valid/test splits.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    download_dir = out_dir / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dataset_slug} from Kaggle...")
    api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True)

    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    rows = []

    if ids is not None:
        labels = _all_label_rows(download_dir)
        sample = labels[labels["id"].isin(ids)].reset_index(drop=True)
        missing = ids - set(sample["id"])
        if missing:
            print(f"WARNING: {len(missing)} requested ids not found in any label CSV: {sorted(missing)[:5]}...")
        for _, row in sample.iterrows():
            match = _find_image_anywhere(download_dir, row["id"])
            if match is None:
                print(f"WARNING: no image file for id={row['id']!r}, skipping")
                continue
            shutil.copy(match, images_out / match.name)
            rows.append({"filename": match.name, "grade": int(row["grade"])})
    else:
        label_csv = _find_label_csv(download_dir)
        print(f"using label csv: {label_csv.relative_to(download_dir)}")
        labels = pd.read_csv(label_csv)
        id_col = "id_code" if "id_code" in labels.columns else labels.columns[0]
        grade_col = "diagnosis" if "diagnosis" in labels.columns else labels.columns[1]
        labels = labels[[id_col, grade_col]].rename(columns={id_col: "id", grade_col: "grade"})

        image_dir = _find_image_dir(download_dir, keyword="train")
        print(f"using image dir: {image_dir.relative_to(download_dir)}")

        sample = labels.sample(n=min(n_samples, len(labels)), random_state=42).reset_index(drop=True)
        for _, row in sample.iterrows():
            matches = list(image_dir.glob(f"{row['id']}.*"))
            if not matches:
                print(f"WARNING: no image file for id={row['id']!r}, skipping")
                continue
            shutil.copy(matches[0], images_out / matches[0].name)
            rows.append({"filename": matches[0].name, "grade": int(row["grade"])})

    if not rows:
        raise RuntimeError("No images were successfully copied -- check the dataset layout")

    pd.DataFrame(rows).to_csv(out_dir / "labels.csv", index=False)
    shutil.rmtree(download_dir)
    print(f"wrote {len(rows)} images + labels to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug (owner/dataset-name)")
    parser.add_argument("--out-dir", default=str(RAW_DIR), help="output directory (default: data/raw/aptos)")
    parser.add_argument("--filenames-csv", default=None, help="CSV with a 'filename' column -- download exactly these ids instead of a random sample")
    args = parser.parse_args()

    ids = None
    if args.filenames_csv:
        filenames = pd.read_csv(args.filenames_csv)["filename"]
        ids = {Path(f).stem for f in filenames}

    download_sample(args.n_samples, args.dataset, out_dir=Path(args.out_dir), ids=ids)


if __name__ == "__main__":
    main()
