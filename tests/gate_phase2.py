"""Phase 2 gate: CORN decode correctness, validation QWK, frozen threshold
existence + untouched test split, and ONNX parity. See docs/BUILD_PLAN.md.

Note: validation QWK here is measured on whatever local checkpoint exists.
A local run on the small APTOS dev sample is real, honestly-computed code --
but AGENTS.md restricts heavy grading-model training to Kaggle notebooks on
the full dataset, so this gate passing locally is not the same claim as the
project's real Phase 2 result.
"""
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from src.drscreen.grading.ordinal import decode_corn

MANIFEST_PATH = Path("models/manifest.json")
RUN_LOG_PATH = Path("models/evaluate_run_log.json")
GRADING_CONFIG = Path("configs/grading.yaml")


def check_corn_decode() -> None:
    BIG = 10.0
    cases = [
        ([-BIG, -BIG, -BIG, -BIG], 0),
        ([BIG, -BIG, -BIG, -BIG], 1),
        ([BIG, BIG, -BIG, -BIG], 2),
        ([BIG, BIG, BIG, -BIG], 3),
        ([BIG, BIG, BIG, BIG], 4),
    ]
    for logit_vals, expected in cases:
        decoded = decode_corn(torch.tensor([logit_vals])).item()
        assert decoded == expected, f"decode_corn({logit_vals}) = {decoded}, expected {expected}"
    print("[ok] CORN decode correctness")


def check_validation_qwk() -> None:
    with open(GRADING_CONFIG) as f:
        config = yaml.safe_load(f)
    checkpoint_path = Path(config.get("checkpoint_dir", "models/grading")) / "best.pt"
    assert checkpoint_path.exists(), f"{checkpoint_path} missing -- run src.drscreen.grading.train first"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    val_qwk = checkpoint["val_qwk"]
    assert val_qwk >= 0.80, f"validation QWK {val_qwk:.4f} < 0.80"
    print(f"[ok] validation QWK = {val_qwk:.4f}")


def check_threshold_frozen_and_test_untouched() -> None:
    assert MANIFEST_PATH.exists(), f"{MANIFEST_PATH} missing -- run scripts/select_threshold.py first"
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert "referable_threshold" in manifest, f"{MANIFEST_PATH} has no frozen referable_threshold"
    assert not RUN_LOG_PATH.exists(), (
        f"{RUN_LOG_PATH} exists -- the test split has already been evaluated; "
        "the threshold must be frozen BEFORE the test split is ever read."
    )
    print(f"[ok] threshold frozen at {manifest['referable_threshold']:.4f}, test split untouched")


def check_onnx_parity() -> None:
    result = subprocess.run([sys.executable, "-m", "scripts.check_parity"], capture_output=True, text=True)
    assert result.returncode == 0, f"scripts.check_parity failed:\n{result.stdout}\n{result.stderr}"
    print("[ok] ONNX parity")


def main() -> None:
    check_corn_decode()
    check_validation_qwk()
    check_threshold_frozen_and_test_untouched()
    check_onnx_parity()
    print("PHASE 2 GATE: PASS")


if __name__ == "__main__":
    main()
