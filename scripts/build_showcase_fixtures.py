"""Precompute full screening results for 6 showcase images spanning the
grade range, so a live demo never waits on a cold pipeline run for its
first impression (see AGENTS.md, Phase 5). Writes JSON matching the API's
/screen response shape (api/main.py's _serialize) into deploy/fixtures/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import _sanitize_nan, _serialize  # reuse the exact API serialization logic
from src.drscreen.pipeline import ScreeningPipeline

MANIFEST = Path("data/cache/aptos_val_manifest.csv")
OUT_DIR = Path("deploy/fixtures")
N_PER_GRADE = 1  # one showcase image per ICDR grade 0-4, plus one more


def pick_showcase_rows(manifest: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for grade in range(5):
        subset = manifest[manifest["grade"] == grade]
        if len(subset) == 0:
            continue
        rows.append(subset.sample(n=1, random_state=int(rng.integers(0, 10_000))))
    picked = pd.concat(rows, ignore_index=True)
    # one extra, to reach 6 -- prefer a referable (grade>=2) case for variety
    remaining = manifest[(manifest["grade"] >= 2) & (~manifest["path"].isin(picked["path"]))]
    if len(remaining) > 0:
        picked = pd.concat([picked, remaining.sample(n=1, random_state=seed)], ignore_index=True)
    return picked


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    showcase_rows = pick_showcase_rows(manifest)

    pipeline = ScreeningPipeline()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for i, row in showcase_rows.iterrows():
        result = pipeline.run(row["path"])
        fixture_id = f"showcase_{i:02d}"
        payload = _sanitize_nan(_serialize(result, session_id=fixture_id))
        payload["_true_grade"] = int(row["grade"])  # for reference only, not part of the API's own response shape
        out_path = OUT_DIR / f"{fixture_id}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        written.append((fixture_id, result.quality_verdict, result.severity_grade))

    print(f"wrote {len(written)} fixtures to {OUT_DIR}")
    for fid, verdict, grade in written:
        print(f"  {fid}: verdict={verdict} grade={grade}")


if __name__ == "__main__":
    main()
