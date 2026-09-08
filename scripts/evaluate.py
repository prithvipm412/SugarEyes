"""Run the frozen model + threshold on Messidor-2 EXACTLY ONCE. Test data is
touched exactly once, at the end (see AGENTS.md) -- this script keeps an
append-only run log and loudly warns on any run after the first.

Requires a "messidor2" dataset already cached with split="test" in
data/cache/manifest.csv, which in turn requires the raw Messidor-2 images
and labels under data/raw/messidor2/ (ADCIS registration -- see
data/splits/README.md). Not available in this environment as of writing.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.drscreen.eval.metrics import (
    bootstrap_ci,
    quadratic_weighted_kappa,
    roc_auc_delong_ci,
    sensitivity_specificity_at_threshold,
)
from src.drscreen.grading.dataset import GradingDataset
from src.drscreen.grading.model import GradingModel
from src.drscreen.grading.ordinal import decode_corn, referable_score

RUN_LOG = Path("models/evaluate_run_log.json")


def _check_run_log() -> list[dict]:
    log = json.loads(RUN_LOG.read_text()) if RUN_LOG.exists() else []
    if log:
        print(f"*** WARNING: scripts/evaluate.py has already been run {len(log)} time(s) on the test split. ***")
        print("*** Test data must be touched exactly once. Do not use this run to pick a different threshold. ***")
        for entry in log:
            print(f"    previous run at {entry['timestamp']}")
    return log


def main() -> None:
    with open("configs/grading.yaml") as f:
        config = yaml.safe_load(f)

    manifest = json.loads(Path("models/manifest.json").read_text())
    threshold = manifest["referable_threshold"]

    device = torch.device("cpu")
    model = GradingModel(backbone_name=config.get("backbone", "resnet34"), pretrained=False).to(device)
    checkpoint_path = Path(config.get("checkpoint_dir", "models/grading")) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ds = GradingDataset(config["manifest_csv"], split="test", image_size=config.get("image_size", 224), train=False)
    loader = DataLoader(ds, batch_size=config.get("batch_size", 16), shuffle=False)

    y_grade, y_referable, y_score = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            y_grade.append(decode_corn(logits).numpy())
            y_referable.append((labels >= 2).numpy())
            y_score.append(referable_score(logits).numpy())
    y_grade = np.concatenate(y_grade)
    y_referable = np.concatenate(y_referable)
    y_score = np.concatenate(y_score)
    y_grade_true = ds.rows["grade"].to_numpy()

    log = _check_run_log()

    sens_point, sens_lo, sens_hi = bootstrap_ci(lambda yt, ys: sensitivity_specificity_at_threshold(yt, ys, threshold)[0], y_referable, y_score)
    spec_point, spec_lo, spec_hi = bootstrap_ci(lambda yt, ys: sensitivity_specificity_at_threshold(yt, ys, threshold)[1], y_referable, y_score)
    auc, auc_lo, auc_hi = roc_auc_delong_ci(y_referable, y_score)
    qwk = quadratic_weighted_kappa(y_grade_true, y_grade)

    print("=== Phase 2 headline metrics (Messidor-2, single evaluation) ===")
    print(f"threshold (frozen from validation): {threshold:.4f}")
    print(f"sensitivity: {sens_point:.4f}  95% CI [{sens_lo:.4f}, {sens_hi:.4f}]  (target >= 0.90)")
    print(f"specificity: {spec_point:.4f}  95% CI [{spec_lo:.4f}, {spec_hi:.4f}]  (target >= 0.85)")
    print(f"AUC (DeLong CI): {auc:.4f}  95% CI [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"quadratic weighted kappa (5-class grade): {qwk:.4f}")

    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold": threshold,
            "sensitivity": sens_point,
            "sensitivity_ci": [sens_lo, sens_hi],
            "specificity": spec_point,
            "specificity_ci": [spec_lo, spec_hi],
            "auc": auc,
            "auc_ci": [auc_lo, auc_hi],
            "qwk": qwk,
        }
    )
    RUN_LOG.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
