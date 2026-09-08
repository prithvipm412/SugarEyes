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


def download_sample(n_samples: int, dataset_slug: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    download_dir = RAW_DIR / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dataset_slug} from Kaggle...")
    api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True)

    label_csvs = list(download_dir.rglob("train*.csv")) + list(download_dir.rglob("*labels*.csv"))
    if not label_csvs:
        raise FileNotFoundError(f"No label CSV found under {download_dir} after download")
    labels = pd.read_csv(label_csvs[0])

    id_col = "id_code" if "id_code" in labels.columns else labels.columns[0]
    grade_col = "diagnosis" if "diagnosis" in labels.columns else labels.columns[1]
    labels = labels[[id_col, grade_col]].rename(columns={id_col: "id", grade_col: "grade"})

    image_dirs = [p for p in download_dir.rglob("*") if p.is_dir() and "image" in p.name.lower()]
    if not image_dirs:
        raise FileNotFoundError(f"No image directory found under {download_dir}")
    image_dir = image_dirs[0]

    sample = labels.sample(n=min(n_samples, len(labels)), random_state=42).reset_index(drop=True)

    images_out = RAW_DIR / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in sample.iterrows():
        matches = list(image_dir.glob(f"{row['id']}.*"))
        if not matches:
            print(f"WARNING: no image file for id={row['id']!r}, skipping")
            continue
        shutil.copy(matches[0], images_out / matches[0].name)
        rows.append({"filename": matches[0].name, "grade": int(row["grade"])})

    if not rows:
        raise RuntimeError("No images were successfully copied -- check the dataset layout")

    pd.DataFrame(rows).to_csv(RAW_DIR / "labels.csv", index=False)
    shutil.rmtree(download_dir)
    print(f"wrote {len(rows)} images + labels to {RAW_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug (owner/dataset-name)")
    args = parser.parse_args()
    download_sample(args.n_samples, args.dataset)


if __name__ == "__main__":
    main()
