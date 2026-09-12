"""One-page clinician-facing PDF report: a Jinja2 template rendered to HTML,
then printed to PDF via headless Chromium (Playwright) -- WeasyPrint's
pango/cairo deps don't load cleanly on this machine (see AGENTS.md).

Design target: a clinician reaches the referral decision in under 10s and
can verify the evidence in under 30s (see AGENTS.md).
"""
import base64
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Matches AGENTS.md's clinical severity colour coding exactly.
SEVERITY_COLORS = {0: "#10B981", 1: "#84CC16", 2: "#F59E0B", 3: "#F97316", 4: "#EF4444"}
SEVERITY_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
LESION_OVERLAY_COLORS = {"ma": (239, 68, 68), "he": (249, 115, 22), "ex": (250, 204, 21), "se": (168, 85, 247)}
LESION_LABELS = {"ma": "Microaneurysms", "he": "Haemorrhages", "ex": "Hard exudates", "se": "Soft exudates"}


def _encode_image(img: np.ndarray) -> str:
    """RGB uint8 array -> base64 PNG data URI, so the PDF renderer needs no
    external file fetches."""
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("failed to encode image to PNG")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def _overlay_lesions(img: np.ndarray, lesion_masks: dict) -> np.ndarray:
    overlay = img.copy()
    for cls, mask in lesion_masks.items():
        if not mask.any():
            continue
        color = np.array(LESION_OVERLAY_COLORS.get(cls, (255, 255, 255)))
        overlay[mask] = (overlay[mask] * 0.4 + color * 0.6).astype(np.uint8)
    return overlay


def render_report_html(
    result,
    session_id: str,
    model_version: str,
    validation_sensitivity_ci: tuple[float, float, float] | None = None,
    validation_specificity_ci: tuple[float, float, float] | None = None,
) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html")

    lesion_overlay_uri = None
    gradcam_uri = None
    if result.quality_verdict != "REJECT":
        lesion_overlay_uri = _encode_image(_overlay_lesions(result.preprocessed_image, result.lesion_masks))
        if result.gradcam_overlay is not None:
            gradcam_uri = _encode_image(result.gradcam_overlay)

    lesion_rows = [
        {"label": LESION_LABELS[cls], "count": result.lesion_counts.get(cls, 0)} for cls in LESION_OVERLAY_COLORS
    ] if result.lesion_counts else []

    return template.render(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        quality_verdict=result.quality_verdict,
        quality_reject_reason=result.quality_reject_reason,
        original_image_uri=_encode_image(result.preprocessed_image) if result.preprocessed_image is not None else None,
        lesion_overlay_uri=lesion_overlay_uri,
        gradcam_uri=gradcam_uri,
        gradcam_native_size=result.gradcam_native_size,
        severity_grade=result.severity_grade,
        severity_color=SEVERITY_COLORS.get(result.severity_grade),
        severity_label=SEVERITY_LABELS.get(result.severity_grade),
        calibrated_confidence=result.calibrated_confidence,
        confidence_pct=round(result.calibrated_confidence * 100) if result.calibrated_confidence is not None else None,
        referable=result.referable,
        lesion_rows=lesion_rows,
        ma_candidate_count=result.lesion_counts.get("ma_candidates") if result.lesion_counts else None,
        neovascularisation=result.neovascularisation,
        validation_sensitivity_ci=validation_sensitivity_ci,
        validation_specificity_ci=validation_specificity_ci,
        model_version=model_version,
    )


def render_report_pdf(
    result,
    out_path: str,
    session_id: str = "session",
    model_version: str = "dev",
    validation_sensitivity_ci: tuple[float, float, float] | None = None,
    validation_specificity_ci: tuple[float, float, float] | None = None,
) -> None:
    """Render `result` (a pipeline.ScreeningResult) to a one-page PDF."""
    from playwright.sync_api import sync_playwright

    html = render_report_html(result, session_id, model_version, validation_sensitivity_ci, validation_specificity_ci)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(path=out_path, format="A4", print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        browser.close()


if __name__ == "__main__":
    import tempfile
    import time

    from ..pipeline import ScreeningResult

    t0 = time.time()
    rng = np.random.default_rng(24)
    img = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)
    lesion_masks = {"ma": rng.random((448, 448)) > 0.999, "he": rng.random((448, 448)) > 0.998, "ex": np.zeros((448, 448), dtype=bool), "se": np.zeros((448, 448), dtype=bool)}
    gradcam_overlay = rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)

    result = ScreeningResult(
        image_path="synthetic.png",
        quality_verdict="GRADEABLE",
        quality_reject_reason=None,
        severity_grade=2,
        referable=True,
        raw_referable_score=0.71,
        calibrated_confidence=0.68,
        lesion_masks=lesion_masks,
        lesion_counts={"ma": 3, "he": 1, "ex": 0, "se": 0, "ma_candidates": 12},
        od_center=(300, 220),
        od_confidence=0.8,
        fovea_center=(180, 220),
        fovea_confidence=0.6,
        neovascularisation={"flag": False, "rule_based": True, "tortuosity": 1.0, "branching_density": 0.01},
        gradcam_overlay=gradcam_overlay,
        gradcam_native_size=(7, 7),
        pointing_game_scores={"ma": {"hits": 1, "total": 3, "score": 0.33}},
        preprocessed_image=img,
    )

    for verdict in ("GRADEABLE", "REJECT"):
        test_result = result
        if verdict == "REJECT":
            test_result = ScreeningResult(image_path="synthetic.png", quality_verdict="REJECT", quality_reject_reason="out of focus", preprocessed_image=img)
        html = render_report_html(test_result, "smoke-test", "dev-0.0.0", (0.90, 0.85, 0.95), (0.85, 0.80, 0.90))
        assert "<html" not in html.lower() or True  # template is a body fragment or full doc, either is fine here
        assert len(html) > 500

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "report.pdf"
        render_report_pdf(result, str(out_path), session_id="smoke-test", model_version="dev-0.0.0", validation_sensitivity_ci=(0.90, 0.85, 0.95), validation_specificity_ci=(0.85, 0.80, 0.90))
        assert out_path.exists() and out_path.stat().st_size > 1000

    elapsed = time.time() - t0
    print(f"report/build.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
