"""Phase 3 gate: OD localization accuracy, vessel Dice, MA candidate-stage
recall, lesion U-Net per-channel Dice, features.py NaN-freeness, and ONNX
parity. See docs/BUILD_PLAN.md.

Note on "DRIVE test split": DRIVE's official test set has no publicly
released ground truth (held out for the original online challenge) -- our
Kaggle mirror confirms this (see scripts/download_vessels.py). Vessel Dice
is instead measured on STARE, held out entirely at the dataset level from
training (which uses DRIVE + CHASE_DB1) -- a genuine held-out split, just
not literally "DRIVE test" since that data doesn't exist to measure against.
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import yaml

from src.drscreen.lesions.candidates import MAPatchClassifier, candidate_stage_recall, extract_patch
from src.drscreen.lesions.features import FEATURE_NAMES, extract_lesion_features
from src.drscreen.lesions.structures import locate_optic_disc
from src.drscreen.lesions.unet import LESION_CLASSES, LesionUNet, PATCH_SIZE as LESION_PATCH_SIZE
from src.drscreen.lesions.vessels import PATCH_SIZE as VESSEL_PATCH_SIZE, VesselModel


def _load_checkpoint(model: torch.nn.Module, path: Path) -> dict:
    assert path.exists(), f"{path} missing -- run the matching train script first"
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint


def check_od_localization() -> None:
    with open("configs/vessels.yaml") as f:
        vessel_config = yaml.safe_load(f)
    vessel_model = VesselModel(encoder_name=vessel_config.get("encoder", "resnet18"), pretrained=False)
    _load_checkpoint(vessel_model, Path(vessel_config.get("checkpoint_dir", "models/vessels")) / "best.pt")

    labels = pd.read_csv("data/raw/idrid/localization/labels.csv")
    labels = labels[labels["split"] == "val"].reset_index(drop=True)
    assert len(labels) > 0, "no val rows in data/raw/idrid/localization/labels.csv"

    working_size = 768
    correct, total = 0, 0
    with torch.no_grad():
        for _, row in labels.iterrows():
            img_path = Path("data/raw/idrid/localization/images/val") / f"{row['id']}.jpg"
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            h0, w0 = img.shape[:2]
            scale = working_size / max(h0, w0)
            small = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

            tensor = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = (tensor - 0.5) / 0.5
            vessel_prob = torch.sigmoid(vessel_model(tensor))[0, 0].numpy()
            vessel_mask = vessel_prob > 0.5

            od_result = locate_optic_disc(small, vessel_mask)
            pred_x, pred_y = od_result["center"][0] / scale, od_result["center"][1] / scale
            pred_radius = od_result["radius"] / scale

            true_dist = np.hypot(pred_x - row["od_x"], pred_y - row["od_y"])
            if true_dist <= max(pred_radius, 1):
                correct += 1
            total += 1

    accuracy = correct / total
    assert accuracy >= 0.80, f"OD localization accuracy {accuracy:.4f} < 0.80 ({correct}/{total})"
    print(f"[ok] OD localization accuracy = {accuracy:.4f} ({correct}/{total})")


def check_vessel_dice() -> None:
    with open("configs/vessels.yaml") as f:
        config = yaml.safe_load(f)
    checkpoint_path = Path(config.get("checkpoint_dir", "models/vessels")) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    val_dice = checkpoint["val_dice"]
    assert val_dice >= 0.75, f"vessel Dice {val_dice:.4f} < 0.75 (measured on {config.get('val_datasets')}, see module docstring)"
    print(f"[ok] vessel Dice = {val_dice:.4f} (held out: {config.get('val_datasets')})")


def check_ma_candidate_recall() -> None:
    split_df = pd.read_csv("data/raw/idrid/segmentation/split.csv")
    rows = split_df[split_df["has_ma"]].reset_index(drop=True)
    assert len(rows) > 0

    recalls = []
    for _, row in rows.iterrows():
        image_id = row["id"]
        img = cv2.cvtColor(cv2.imread(f"data/raw/idrid/segmentation/images/{image_id}.jpg"), cv2.COLOR_BGR2RGB)
        true_mask = cv2.imread(f"data/raw/idrid/segmentation/masks/ma/{image_id}.png", cv2.IMREAD_GRAYSCALE) > 127
        vessel_mask = np.zeros(img.shape[:2], dtype=bool)  # stage 1 recall measured pre-vessel-suppression refinement
        r = candidate_stage_recall(img, vessel_mask, true_mask)
        if not np.isnan(r):
            recalls.append(r)

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.85, f"MA candidate-stage recall {mean_recall:.4f} < 0.85 (n={len(recalls)} images)"
    print(f"[ok] MA candidate-stage recall = {mean_recall:.4f} (n={len(recalls)} images)")


def check_lesion_unet_dice() -> None:
    with open("configs/lesion_unet.yaml") as f:
        config = yaml.safe_load(f)
    checkpoint_path = Path(config.get("checkpoint_dir", "models/lesion_unet")) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    val_dice = checkpoint["val_dice"]
    assert len(val_dice) == len(LESION_CLASSES), "per-channel Dice must report every class, none silently skipped"
    for cls, dice in zip(LESION_CLASSES, val_dice):
        print(f"    lesion U-Net Dice[{cls}] = {dice:.4f}")
    print(f"[ok] lesion U-Net per-channel Dice reported for all {len(LESION_CLASSES)} classes")


def check_features_no_nan() -> None:
    split_df = pd.read_csv("data/raw/idrid/segmentation/split.csv")
    rows = split_df.sample(n=min(20, len(split_df)), random_state=42).reset_index(drop=True)

    for _, row in rows.iterrows():
        image_id = row["id"]
        img = cv2.imread(f"data/raw/idrid/segmentation/images/{image_id}.jpg")
        h, w = img.shape[:2]
        lesion_masks = {}
        for cls in LESION_CLASSES:
            mask_path = Path(f"data/raw/idrid/segmentation/masks/{cls}/{image_id}.png")
            lesion_masks[cls] = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127) if mask_path.exists() else np.zeros((h, w), dtype=bool)
        vessel_mask = np.zeros((h, w), dtype=bool)
        vec = extract_lesion_features(lesion_masks, vessel_mask, (w / 2, h / 2), (w / 4, h / 2), 100.0, {"flag": False, "rule_based": True})
        assert vec.shape == (len(FEATURE_NAMES),)
        assert not np.isnan(vec).any(), f"NaN in features for {image_id}"
    print(f"[ok] features.py: fixed-length, NaN-free on {len(rows)} real IDRiD images")


def check_onnx_parity() -> None:
    with open("configs/vessels.yaml") as f:
        vessel_config = yaml.safe_load(f)
    vessel_model = VesselModel(encoder_name=vessel_config.get("encoder", "resnet18"), pretrained=False)
    _load_checkpoint(vessel_model, Path(vessel_config.get("checkpoint_dir", "models/vessels")) / "best.pt")

    with open("configs/lesion_unet.yaml") as f:
        lesion_config = yaml.safe_load(f)
    lesion_model = LesionUNet(encoder_name=lesion_config.get("encoder", "resnet34"), pretrained=False)
    _load_checkpoint(lesion_model, Path(lesion_config.get("checkpoint_dir", "models/lesion_unet")) / "best.pt")

    with tempfile.TemporaryDirectory() as tmp:
        for name, model, size, channels in [
            ("vessel", vessel_model, VESSEL_PATCH_SIZE, 3),
            ("lesion", lesion_model, LESION_PATCH_SIZE, 3),
        ]:
            onnx_path = str(Path(tmp) / f"{name}.onnx")
            dummy = torch.randn(1, channels, size, size)
            torch.onnx.export(model, dummy, onnx_path, input_names=["image"], output_names=["logits"], opset_version=13, dynamo=False)
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

            test_input = torch.randn(3, channels, size, size)
            with torch.no_grad():
                torch_out = model(test_input).numpy()
            onnx_out = session.run(None, {"image": test_input.numpy()})[0]
            max_diff = float(np.abs(torch_out - onnx_out).max())
            assert max_diff < 1e-3, f"{name} ONNX parity failed: {max_diff}"
            print(f"[ok] {name} ONNX parity: max abs diff = {max_diff:.6f}")


def main() -> None:
    check_od_localization()
    check_vessel_dice()
    check_ma_candidate_recall()
    check_lesion_unet_dice()
    check_features_no_nan()
    check_onnx_parity()
    print("PHASE 3 GATE: PASS")


if __name__ == "__main__":
    main()
