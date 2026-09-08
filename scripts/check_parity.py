"""Verify PyTorch and ONNX Runtime agree on the grading model's outputs.
Preprocessing must be byte-identical on both paths -- this is what makes the
ONNX export trustworthy for local CPU/MPS deployment (see AGENTS.md)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import yaml

from src.drscreen.grading.model import GradingModel, export_onnx

N_IMAGES = 50
MAX_ABS_DIFF = 1e-3


def main() -> None:
    with open("configs/grading.yaml") as f:
        config = yaml.safe_load(f)
    image_size = config.get("image_size", 224)

    checkpoint_path = Path(config.get("checkpoint_dir", "models/grading")) / "best.pt"
    assert checkpoint_path.exists(), f"{checkpoint_path} missing -- run src.drscreen.grading.train first"

    device = torch.device("cpu")
    model = GradingModel(backbone_name=config.get("backbone", "resnet34"), pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = str(Path(tmp) / "grading.onnx")
        export_onnx(model, onnx_path, image_size=image_size)
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        manifest = pd.read_csv(config["manifest_csv"])
        sample = manifest.sample(n=min(N_IMAGES, len(manifest)), random_state=42)

        max_diff = 0.0
        with torch.no_grad():
            for _, row in sample.iterrows():
                img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
                tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                tensor = (tensor - 0.5) / 0.5

                torch_logits = model(tensor).numpy()
                onnx_logits = session.run(None, {"image": tensor.numpy()})[0]
                max_diff = max(max_diff, float(np.abs(torch_logits - onnx_logits).max()))

    print(f"max abs logit difference over {len(sample)} images: {max_diff:.6f}")
    assert max_diff < MAX_ABS_DIFF, f"parity check failed: {max_diff} >= {MAX_ABS_DIFF}"
    print("PARITY CHECK: PASS")


if __name__ == "__main__":
    main()
