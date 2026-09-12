"""End-to-end orchestrator: one function, one image in, one result object out.

Loads every model once (see AGENTS.md -- ONNX/checkpoints loaded once, not
per call) and exposes both the full quality-gated flow (`ScreeningPipeline.run`,
for a real end-user image) and a lighter grading+lesion-features path
(`ScreeningPipeline.grade_and_extract_features`, for building the fusion
training set from already-curated, already-gradeable images where the
quality gate isn't the point).

All Phase 3 lesion models run on the same 448x448 working resolution the
grading model uses (matching AGENTS.md's cache size), not IDRiD's native
~4288px. This is a real, honest trade-off: the MA cascade's measured 96%
recall in Phase 3 was on native-resolution IDRiD images -- at 448px there
are fewer pixels for the top-hat threshold to work with, so recall here is
expected to be lower. That's reported, not hidden, the same way IDRiD's
soft-exudate and microaneurysm Dice scores are.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .device import get_device
from .explain.calibrate import apply_temperature
from .explain.gradcam import compute_gradcam
from .explain.pointing import pointing_game_per_class
from .grading.model import GradingModel
from .grading.ordinal import decode_corn, referable_score
from .lesions.candidates import MAPatchClassifier, extract_candidates
from .lesions.features import extract_lesion_features
from .lesions.neovascularisation import detect_neovascularisation
from .lesions.structures import locate_fovea, locate_optic_disc
from .lesions.unet import LESION_CLASSES, LesionUNet
from .lesions.vessels import VesselModel
from .preprocess.retina import preprocess_image
from .quality.features import extract_features as extract_quality_features
from .quality.model import CLASS_NAMES as QUALITY_CLASS_NAMES
from .quality.model import IQAModel, reject_reason

CACHE_SIZE = 448
GRADING_INPUT_SIZE = 224


@dataclass
class ScreeningResult:
    image_path: str
    quality_verdict: str
    quality_reject_reason: str | None
    severity_grade: int | None = None
    referable: bool | None = None
    raw_referable_score: float | None = None
    calibrated_confidence: float | None = None
    lesion_masks: dict[str, np.ndarray] = field(default_factory=dict)
    lesion_counts: dict[str, int] = field(default_factory=dict)
    od_center: tuple[int, int] | None = None
    od_confidence: float | None = None
    fovea_center: tuple[int, int] | None = None
    fovea_confidence: float | None = None
    neovascularisation: dict | None = None
    gradcam_overlay: np.ndarray | None = None
    gradcam_native_size: tuple[int, int] | None = None
    pointing_game_scores: dict | None = None
    preprocessed_image: np.ndarray | None = None


def _load_manifest() -> dict:
    path = Path("models/manifest.json")
    return json.loads(path.read_text()) if path.exists() else {}


class ScreeningPipeline:
    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()
        manifest = _load_manifest()
        self.referable_threshold = manifest.get("referable_threshold")
        self.calibration_temperature = manifest.get("calibration_temperature", 1.0)

        with open("configs/iqa.yaml") as f:
            iqa_config = yaml.safe_load(f)
        self.iqa_model = IQAModel(backbone_name=iqa_config.get("backbone", "resnet18"), pretrained=False).to(self.device)
        self._load(self.iqa_model, Path(iqa_config.get("checkpoint_dir", "models/iqa")) / "best.pt")
        stats_path = Path(iqa_config.get("checkpoint_dir", "models/iqa")) / "feature_stats.json"
        self.iqa_feature_stats = json.loads(stats_path.read_text()) if stats_path.exists() else None

        with open("configs/grading.yaml") as f:
            grading_config = yaml.safe_load(f)
        self.grading_model = GradingModel(backbone_name=grading_config.get("backbone", "resnet34"), pretrained=False).to(self.device)
        self._load(self.grading_model, Path(grading_config.get("checkpoint_dir", "models/grading")) / "best.pt")
        self.grading_model.eval()

        with open("configs/vessels.yaml") as f:
            vessel_config = yaml.safe_load(f)
        self.vessel_model = VesselModel(encoder_name=vessel_config.get("encoder", "resnet18"), pretrained=False).to(self.device)
        self._load(self.vessel_model, Path(vessel_config.get("checkpoint_dir", "models/vessels")) / "best.pt")
        self.vessel_model.eval()

        with open("configs/lesion_unet.yaml") as f:
            lesion_config = yaml.safe_load(f)
        self.lesion_model = LesionUNet(encoder_name=lesion_config.get("encoder", "resnet34"), pretrained=False).to(self.device)
        self._load(self.lesion_model, Path(lesion_config.get("checkpoint_dir", "models/lesion_unet")) / "best.pt")
        self.lesion_model.eval()

        ma_ckpt_path = Path("models/ma_classifier/best.pt")
        self.ma_classifier = MAPatchClassifier().to(self.device)
        if ma_ckpt_path.exists():
            self._load(self.ma_classifier, ma_ckpt_path)
        self.ma_classifier.eval()

    def _load(self, model: torch.nn.Module, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{checkpoint_path} missing -- train this model before running the pipeline")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state"])

    def _to_tensor(self, img: np.ndarray, size: int) -> torch.Tensor:
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4) if img.shape[:2] != (size, size) else img
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return ((tensor - 0.5) / 0.5).to(self.device)

    def _check_quality(self, img_448: np.ndarray, mask: np.ndarray) -> tuple[str, str | None]:
        features = extract_quality_features(img_448, mask)
        tensor = self._to_tensor(img_448, 224)
        with torch.no_grad():
            logits = self.iqa_model(tensor, torch.from_numpy(features).unsqueeze(0).to(self.device))
        verdict = QUALITY_CLASS_NAMES[int(logits.argmax(dim=1).item())]
        reason = None
        if verdict == "REJECT" and self.iqa_feature_stats is not None:
            reason = reject_reason(features, self.iqa_feature_stats)
        return verdict, reason

    def _run_lesions(self, img_448: np.ndarray) -> dict:
        """The Phase 3 sub-pipeline: vessels, OD/fovea, 4-class lesion
        masks, MA candidates, NV heuristic, and the fusion feature vector."""
        with torch.no_grad():
            vessel_prob = torch.sigmoid(self.vessel_model(self._to_tensor(img_448, img_448.shape[0])))[0, 0]
        vessel_mask = (vessel_prob > 0.5).cpu().numpy()

        od_result = locate_optic_disc(img_448, vessel_mask)
        fovea_result = locate_fovea(img_448, od_result["center"], od_result["radius"], vessel_mask)

        with torch.no_grad():
            lesion_logits = self.lesion_model(self._to_tensor(img_448, img_448.shape[0]))
        lesion_probs = torch.sigmoid(lesion_logits)[0].cpu().numpy()
        lesion_masks = {cls: lesion_probs[i] > 0.5 for i, cls in enumerate(LESION_CLASSES)}

        ma_candidates = extract_candidates(img_448, vessel_mask)
        if ma_candidates:
            from .lesions.candidates import extract_patch, PATCH_SIZE as MA_PATCH_SIZE

            patches = np.stack([extract_patch(img_448, c["center"], MA_PATCH_SIZE) for c in ma_candidates])
            patch_tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float().to(self.device) / 255.0
            patch_tensor = (patch_tensor - 0.5) / 0.5
            with torch.no_grad():
                ma_scores = torch.sigmoid(self.ma_classifier(patch_tensor)).cpu().numpy()
            confirmed_ma = (ma_scores > 0.5).sum()
        else:
            confirmed_ma = 0

        nv_result = detect_neovascularisation(vessel_mask, od_result["center"], od_result["radius"])

        feature_vector = extract_lesion_features(
            lesion_masks, vessel_mask, fovea_result["center"], od_result["center"], od_result["radius"], nv_result
        )

        return {
            "lesion_masks": lesion_masks,
            "lesion_counts": {cls: _count_components(m) for cls, m in lesion_masks.items()} | {"ma_candidates": int(confirmed_ma)},
            "vessel_mask": vessel_mask,
            "od_result": od_result,
            "fovea_result": fovea_result,
            "nv_result": nv_result,
            "feature_vector": feature_vector,
        }

    def grade_and_extract_features(self, img_448: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Grading logits + lesion feature vector + raw referable score +
        decoded grade for an already-preprocessed 448x448 image -- skips the
        quality gate and Grad-CAM, for building the fusion training set."""
        with torch.no_grad():
            grading_logits = self.grading_model(self._to_tensor(img_448, GRADING_INPUT_SIZE))
        grade = int(decode_corn(grading_logits)[0].item())
        raw_score = float(referable_score(grading_logits)[0].item())
        lesion_result = self._run_lesions(img_448)
        return grading_logits[0].detach().cpu().numpy(), lesion_result["feature_vector"], raw_score, grade

    def run(self, image_path: str) -> ScreeningResult:
        raw = cv2.imread(image_path)
        if raw is None:
            raise FileNotFoundError(f"could not read image: {image_path}")
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img_448, mask = preprocess_image(raw, size=CACHE_SIZE)

        verdict, reject_msg = self._check_quality(img_448, mask)
        result = ScreeningResult(image_path=image_path, quality_verdict=verdict, quality_reject_reason=reject_msg, preprocessed_image=img_448)
        if verdict == "REJECT":
            return result

        with torch.no_grad():
            grading_tensor = self._to_tensor(img_448, GRADING_INPUT_SIZE)
            grading_logits = self.grading_model(grading_tensor)
        result.severity_grade = int(decode_corn(grading_logits)[0].item())
        raw_score = float(referable_score(grading_logits)[0].item())
        result.raw_referable_score = raw_score
        result.calibrated_confidence = float(apply_temperature(np.array([raw_score]), self.calibration_temperature)[0])
        result.referable = raw_score >= self.referable_threshold if self.referable_threshold is not None else result.severity_grade >= 2

        lesion_result = self._run_lesions(img_448)
        result.lesion_masks = lesion_result["lesion_masks"]
        result.lesion_counts = lesion_result["lesion_counts"]
        result.od_center = lesion_result["od_result"]["center"]
        result.od_confidence = lesion_result["od_result"]["confidence"]
        result.fovea_center = lesion_result["fovea_result"]["center"]
        result.fovea_confidence = lesion_result["fovea_result"]["confidence"]
        result.neovascularisation = lesion_result["nv_result"]

        cam_result = compute_gradcam(self.grading_model, grading_tensor)
        result.gradcam_overlay = cam_result["overlay"]
        result.gradcam_native_size = cam_result["native_size"]
        cam_resized_to_448 = cv2.resize(cam_result["cam_upsampled"], (CACHE_SIZE, CACHE_SIZE))
        result.pointing_game_scores = pointing_game_per_class(cam_resized_to_448, result.lesion_masks)

        return result


def _count_components(mask: np.ndarray) -> int:
    from scipy import ndimage

    _labeled, n = ndimage.label(mask.astype(bool))
    return int(n)


def run_pipeline(image_path: str, pipeline: ScreeningPipeline | None = None) -> ScreeningResult:
    """Convenience entry point: one image in, one ScreeningResult out. Pass
    a pre-built `pipeline` when processing many images (model loading is
    the expensive part -- see AGENTS.md, load once, not per call)."""
    pipeline = pipeline or ScreeningPipeline()
    return pipeline.run(image_path)


if __name__ == "__main__":
    import time

    import pandas as pd

    manifest_path = Path("data/cache/manifest.csv")
    if not manifest_path.exists():
        print("SKIP: data/cache/manifest.csv not found -- run Phase 1's build_cache first")
    else:
        t0 = time.time()
        pipeline = ScreeningPipeline()
        manifest = pd.read_csv(manifest_path)
        sample = manifest.sample(n=min(5, len(manifest)), random_state=42)

        for _, row in sample.iterrows():
            result = pipeline.run(row["path"])
            assert result.quality_verdict in ("GRADEABLE", "ENHANCEABLE", "REJECT")
            if result.quality_verdict != "REJECT":
                assert result.severity_grade in range(5)
                assert 0.0 <= result.calibrated_confidence <= 1.0
                assert set(result.lesion_masks) == set(LESION_CLASSES)
                assert result.gradcam_overlay is not None

        elapsed = time.time() - t0
        print(f"pipeline.py smoke test ok in {elapsed:.3f}s for {len(sample)} images ({(elapsed)/len(sample):.2f}s/image incl. model load)")
        assert elapsed < 60
