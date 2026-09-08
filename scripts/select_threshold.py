"""Select the referable-DR decision threshold on the VALIDATION split only,
then freeze it to models/manifest.json. Refuses to run against a test split
-- threshold tuning on Messidor-2 would be test-set leakage (see AGENTS.md:
threshold is selected on validation only, then frozen before test evaluation).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.drscreen.eval.metrics import sensitivity_specificity_at_threshold
from src.drscreen.grading.dataset import GradingDataset
from src.drscreen.grading.model import GradingModel
from src.drscreen.grading.ordinal import referable_score

FORBIDDEN_SPLIT_NAMES = {"test", "messidor2_test", "messidor2"}


def select_threshold(config: dict, checkpoint_path: str, split: str = "val") -> dict:
    assert split not in FORBIDDEN_SPLIT_NAMES, (
        f"select_threshold refuses to run against split={split!r} -- the threshold must be "
        "chosen on validation only, then frozen before the test split is ever read."
    )

    device = torch.device("cpu")
    model = GradingModel(backbone_name=config.get("backbone", "resnet34"), pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ds = GradingDataset(config["manifest_csv"], split=split, image_size=config.get("image_size", 224), train=False)
    loader = DataLoader(ds, batch_size=config.get("batch_size", 16), shuffle=False)

    y_true, y_score = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            y_true.append((labels >= 2).numpy())
            y_score.append(referable_score(logits).numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    best = {"threshold": None, "sensitivity": -1.0, "specificity": -1.0}
    for threshold in np.linspace(0.01, 0.99, 197):
        sens, spec = sensitivity_specificity_at_threshold(y_true, y_score, threshold)
        if sens >= 0.90 and spec > best["specificity"]:
            best = {"threshold": float(threshold), "sensitivity": float(sens), "specificity": float(spec)}

    if best["threshold"] is None:
        raise RuntimeError(
            "No threshold on the validation split reaches sensitivity >= 0.90 -- "
            "the model needs to improve before a threshold can be selected."
        )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    checkpoint_path = args.checkpoint or str(Path(config.get("checkpoint_dir", "models/grading")) / "best.pt")

    result = select_threshold(config, checkpoint_path, split=args.split)
    print(f"selected threshold={result['threshold']:.4f}  sensitivity={result['sensitivity']:.4f}  specificity={result['specificity']:.4f}")

    manifest_path = Path("models/manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["referable_threshold"] = result["threshold"]
    manifest["referable_threshold_val_sensitivity"] = result["sensitivity"]
    manifest["referable_threshold_val_specificity"] = result["specificity"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote frozen threshold to {manifest_path}")


if __name__ == "__main__":
    main()
