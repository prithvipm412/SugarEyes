"""Phase 1 gate: cache round-trip, enhance mask integrity, IQA validation
accuracy, and end-to-end preprocess+IQA latency. See docs/BUILD_PLAN.md."""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.drscreen.preprocess import enhance
from src.drscreen.quality.dataset import IQADataset
from src.drscreen.quality.features import extract_features
from src.drscreen.quality.model import IQAModel
from src.drscreen.preprocess.retina import cached_fov_mask, preprocess_image

MANIFEST = Path("data/cache/manifest.csv")
IQA_CONFIG = Path("configs/iqa.yaml")
IQA_CHECKPOINT = Path("models/iqa/best.pt")


def _load_config() -> dict:
    with open(IQA_CONFIG) as f:
        return yaml.safe_load(f)


def _load_model(config: dict) -> IQAModel:
    assert IQA_CHECKPOINT.exists(), f"{IQA_CHECKPOINT} missing -- run `python -m src.drscreen.quality.train` first"
    device = torch.device("cpu")
    model = IQAModel(backbone_name=config.get("backbone", "resnet18"), pretrained=False).to(device)
    checkpoint = torch.load(IQA_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def check_cache_roundtrip() -> None:
    manifest = pd.read_csv(MANIFEST)
    assert len(manifest) > 0, f"{MANIFEST} is empty -- run scripts/build_cache.py first"
    row = manifest.iloc[0]
    cached = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)

    raw_path = Path("data/raw") / row["dataset"] / "images" / Path(row["path"]).name
    raw = cv2.cvtColor(cv2.imread(str(raw_path)), cv2.COLOR_BGR2RGB)
    recomputed, _ = preprocess_image(raw, size=cached.shape[0])
    assert np.array_equal(cached, recomputed), "cached image does not match a fresh preprocess of the raw source"
    print("[ok] cache round-trip")


def check_enhance_mask_integrity() -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
    mask = cached_fov_mask(448)

    for fn, kwargs in [
        (enhance.clahe_green, {}),
        (enhance.clahe_green, {"mode": "lab"}),
        (enhance.illumination_normalize, {}),
        (enhance.denoise, {}),
    ]:
        out = fn(img, mask, **kwargs)
        assert np.array_equal(out[~mask], img[~mask]), f"{fn.__name__}{kwargs} modified pixels outside the FOV mask"
    print("[ok] enhance mask integrity")


def check_iqa_accuracy() -> None:
    config = _load_config()
    model = _load_model(config)

    val_ds = IQADataset(config["manifest_csv"], split="val", image_size=config.get("image_size", 224))
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 16), shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for images, features, labels in val_loader:
            logits = model(images, features)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    assert acc >= 0.80, f"IQA validation accuracy {acc:.4f} < 0.80"
    print(f"[ok] IQA validation accuracy = {acc:.4f}")


def check_latency() -> None:
    config = _load_config()
    model = _load_model(config)
    image_size = config.get("image_size", 224)

    manifest = pd.read_csv(MANIFEST)
    row = manifest.iloc[0]
    raw_path = Path("data/raw") / row["dataset"] / "images" / Path(row["path"]).name
    raw = cv2.cvtColor(cv2.imread(str(raw_path)), cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    img, mask = preprocess_image(raw, size=448)
    feats = extract_features(img, mask)
    resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        model(tensor, torch.from_numpy(feats).unsqueeze(0))
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, f"preprocess+IQA latency {elapsed:.3f}s >= 1.5s"
    print(f"[ok] preprocess+IQA latency = {elapsed:.3f}s")


def main() -> None:
    check_cache_roundtrip()
    check_enhance_mask_integrity()
    check_iqa_accuracy()
    check_latency()
    print("PHASE 1 GATE: PASS")


if __name__ == "__main__":
    main()
