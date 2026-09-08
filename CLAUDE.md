# Project: Explainable AI for Diabetic Retinopathy Screening

## What this is
A screening pipeline that takes a colour fundus photograph and returns:
1. An image-quality verdict (gradeable / enhance-then-grade / reject with a recapture reason)
2. An International Clinical DR severity grade, 0–4
3. A referable-DR decision (grade >= 2) with a calibrated confidence
4. Lesion evidence: microaneurysms, haemorrhages, hard exudates, soft exudates, vessels, optic disc, fovea
5. A Grad-CAM attention map, with a quantified agreement score against the lesion masks
6. A one-page annotated PDF report designed to be read by a clinician in under 30 seconds

Plus a discrete-event simulation of a district screening programme serving 100,000+ patients a year.

Built for Smart India Hackathon 2026, problem statement 26038. Judged by MathWorks engineers.

## Hard performance targets (from the problem statement)
- Referable DR (grade >= 2): **sensitivity >= 90%**, **specificity >= 85%**
- Reported on Messidor-2 as a held-out test set, with bootstrap 95% confidence intervals
- The integrated pipeline must be shown to outperform any single-technique approach (ablation table)

## Hardware and compute constraints — these are not negotiable
- Development machine: **Apple M4, 16 GB unified memory, no CUDA**
- Usable RAM headroom is ~8 GB after macOS. Assume it.
- **All heavy training happens on Kaggle notebooks** (free T4 x2 or P100, 30 h/week, 12 h session cap).
  APTOS 2019 and EyePACS are already hosted on Kaggle — never download them locally.
- Local MPS training is permitted **only** for the IQA CNN and the MA patch classifier (both small).
- Local inference is CPU/MPS via ONNX Runtime. Budget under 6 seconds for the full pipeline per image.
- Never write code that assumes `cuda`. Use a `get_device()` helper that resolves cuda -> mps -> cpu.

## Tech stack — do not substitute without asking
- Python 3.11, PyTorch 2.x, `timm`, `segmentation-models-pytorch`, `albumentations`
- OpenCV (`opencv-python-headless`), scikit-image, NumPy, SciPy, pandas, scikit-learn
- `grad-cam` (jacobgil/pytorch-grad-cam) for explainability
- `onnx` + `onnxruntime` for deployment
- **SimPy** for the discrete-event screening simulation
- FastAPI + uvicorn for the API
- Vite + React + TypeScript + Tailwind + shadcn/ui + Framer Motion + Recharts for the UI
- Jinja2 + WeasyPrint for PDF reports (fall back to Playwright headless print if WeasyPrint's
  pango/cairo deps are painful on macOS)
- TensorBoard for training curves. No paid tracking services.

## Repository layout
```
configs/          YAML configs, one per trainable model. No hyperparameters in code.
data/raw/         Downloaded datasets. Gitignored.
data/cache/       Preprocessed uint8 images. Gitignored.
data/splits/      CSV split manifests. COMMITTED — these are reproducibility artifacts.
src/drscreen/
  preprocess/     retina.py (crop/mask/resize), enhance.py (CLAHE/illum/denoise), degrade.py (camera sim)
  quality/        features.py (focus/illum/FOV metrics), model.py, train.py
  grading/        model.py, ordinal.py (CORN loss + decode), train.py
  lesions/        unet.py, candidates.py (MA top-hat), vessels.py, structures.py (OD/fovea)
  explain/        gradcam.py, pointing.py (pointing-game metric), calibrate.py (temperature scaling)
  fusion/         model.py
  eval/           metrics.py, ablation.py
  report/         build.py, templates/
  sim/            screening_des.py (SimPy)
  pipeline.py     End-to-end orchestrator. One function, one image in, one result object out.
api/main.py       FastAPI app
web/              Vite React frontend
notebooks/        Kaggle training notebooks
matlab/           Phase 6 compliance layer
models/           ONNX + checkpoints. Gitignored except manifest.json
tests/
```

## Design system (the UI is a judged deliverable — treat it as such)
- Deep navy base: `#0A1628` background, `#0F2440` surfaces, `#1B3A5F` borders
- Accent: `#3B82F6` primary, `#22D3EE` highlights
- Clinical severity colours: green `#10B981` (0), lime `#84CC16` (1), amber `#F59E0B` (2),
  orange `#F97316` (3), red `#EF4444` (4)
- Inter or Geist for UI text, JetBrains Mono for numbers and metrics
- Framer Motion on every state transition. Nothing should pop into existence.
- Grad-CAM overlays: opacity slider, side-by-side original vs heatmap, not a fixed blend
- Dark theme only. This is a clinical tool viewed in dim rooms.

## Engineering rules
1. **Never fabricate a metric.** If a number isn't computed from real data by real code, it does not
   go in a README, a log line, a slide, or a docstring. Print "NOT YET MEASURED" instead.
2. **Never silently fall back.** If pretrained weights fail to download, raise. Do not initialise
   randomly and continue. If a file is missing, raise. Silent fallbacks are how a whole pipeline
   ends up training on noise.
3. **Fix seeds everywhere.** `torch.manual_seed`, `np.random.seed`, `random.seed`, and
   `torch.use_deterministic_algorithms(True)` where it doesn't kill throughput.
4. **Test data is touched exactly once, at the end.** Messidor-2 is never used for threshold
   selection, early stopping, hyperparameter choice, or "just checking."
5. **Checkpoint every epoch.** Kaggle sessions die at 12 hours without warning.
6. **All paths come from config or CLI args.** No hardcoded `/Users/...` anywhere.
7. **Every module gets a `if __name__ == "__main__"` smoke test** that runs on 5 images in
   under 30 seconds.
8. Type hints on every public function. Docstrings that say what, not how.

## Known risks — mitigations are mandatory, not optional
| Risk | Required mitigation |
|---|---|
| Grad-CAM at 14x14 cannot localise 15-pixel microaneurysms | Use a **pointing-game** metric (fraction of annotated lesions inside the top-10% CAM activation), never IoU. State the resolution limit explicitly in the report. |
| APTOS -> Messidor-2 domain shift and label mismatch | Document the exact Messidor-2 label file and grade mapping used in `data/splits/README.md`. Expect and report a performance drop. |
| IDRiD has only 81 pixel-annotated images | Patch-based training with heavy augmentation. Report soft-exudate performance separately and honestly — it will be weak. |
| Neovascularisation has no public pixel ground truth | Ship a rule-based vessel-tortuosity heuristic. Label it as rule-based in the UI and the report. Do not claim it is learned. |
| Fusion may not beat CNN-only | Report the ablation table as measured either way. If fusion loses on AUC, pivot the claim to calibration and interpretability — do not hide the result. |
| Threshold tuning is test-set leakage if done on Messidor-2 | Three-way split. Threshold is selected on validation only, then frozen before test evaluation. |
| No fundus camera, so field conditions are synthetic | The degradation simulator is a stated limitation. Say so in the slides. Do not imply real field validation. |
| No ophthalmologist for the "clinically useful" rating | Build the rating harness anyway (a simple 5-point form over 30 cases). Getting one resident to fill it is the highest-value single action in this project. |

## What "done" means for any task
The code runs, the gate command passes, the numbers printed are real, and a commit exists.
