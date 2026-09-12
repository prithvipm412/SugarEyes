# SugarEyes — Explainable DR Screening

An open-source screening pipeline for diabetic retinopathy: given a colour fundus
photograph, it produces an image-quality verdict, an International Clinical DR
severity grade (0-4), a referable-DR decision with calibrated confidence, lesion
evidence (microaneurysms, haemorrhages, exudates, vessels, optic disc, fovea), a
Grad-CAM explanation with a quantified pointing-game agreement score, and a one-page
clinician-facing PDF report. It also includes a discrete-event simulation of a
district screening programme.

Built for Smart India Hackathon 2026, problem statement 26038.

> **Not a certified medical device.** This is a hackathon research prototype. It
> has not been through regulatory clearance (e.g. CDSCO, FDA) or prospective
> clinical validation, and the "field conditions" it's tested under are a
> synthetic degradation simulator, not a real camera in a real clinic — see
> the known-risks table below. It is not intended for, and must not be used
> for, real patient diagnosis or treatment decisions.

## Project context

- **`AGENTS.md`** / **`CLAUDE.md`** (identical) — persistent project context: hard
  performance targets, hardware/compute constraints, tech stack, repository layout,
  design system, engineering rules, and known risks with required mitigations. Read
  automatically by Claude Code / Antigravity on every turn.
- **`docs/BUILD_PLAN.md`** — the phase-by-phase build plan (Phases 1-6) and their gate
  commands. Phase 2 is the go/no-go: if its sensitivity/specificity targets aren't
  met, nothing after it matters.

## Status

- [x] Phase 0 — repo scaffolding, `AGENTS.md`/`CLAUDE.md`, directory layout
- [x] Phase 1 — data spine, preprocessing, quality gate. Real gate pass on 300 APTOS
      images: IQA validation accuracy 0.8370, latency 0.098s.
- [x] Phase 2 — CORN ordinal grading model, trained for real on Kaggle's full
      ~3662-image APTOS set (`notebooks/kaggle_train_grading.ipynb`, GPU T4x2):
      validation QWK 0.8744, threshold frozen at 0.395 (val sensitivity 0.9058 /
      specificity 0.9480 — already past both PS targets on APTOS's own held-out
      split), ONNX parity 1.7e-5. **The official Messidor-2 test-set result
      (see below) does not meet the sensitivity target** — reported honestly,
      not hidden, per AGENTS.md's core rule.
- [x] Phase 3 — retinal structures and lesions. Real gate pass on real
      DRIVE/CHASE_DB1/STARE (vessels) + IDRiD (lesions/localization) data:
      OD localization 0.9320 (96/103), vessel Dice 0.7940 (held out on STARE),
      MA candidate-stage recall 0.9638, lesion U-Net per-channel Dice
      (ma=0.410 he=0.603 ex=0.783 se=0.569, all reported), ONNX parity
      1.7e-5/2.1e-5. Unlike Phase 2, no dataset here needed registration —
      all four came from Kaggle mirrors.
- [x] Phase 4 — explainability, calibration, fusion, reports. Real gate pass
      against the real Kaggle-trained grading model + the real 550-image
      validation set: Grad-CAM runs on the exported model (native grid 7x7),
      pointing-game reported per class (ma=0.19 he=0.11 ex=0.30 se=0.08, none
      skipped), ECE 0.0386→0.0374 after temperature scaling, PDF report in
      1.24s. Ablation table: **fusion does not beat CNN-only** (0.975 vs
      0.967/0.972 AUC) — reported as measured, exactly the risk AGENTS.md
      flagged in advance, not hidden. New `pipeline.py` orchestrates Phases
      1-4 end to end.
- [x] Phase 5 — screening simulation, API, and UI. Real gate pass: 1000-patient
      SimPy simulation in 0.017s with exact queue conservation (in=out), FastAPI
      `/screen` round-trip on a real image in 0.18-0.87s, frontend (Vite + React +
      TypeScript + Tailwind + shadcn/ui) builds clean with zero TypeScript errors.
      The auto-clear-threshold trade-off — the centrepiece chart for the slides —
      is driven by real (grade, calibrated confidence) pairs resampled from Phase
      4's actual validation-set output, not a synthetic error assumption
      (`results/simulation/threshold_tradeoff.{csv,html}`). Full backend+frontend
      verified end-to-end with a real running server and headless-browser
      interaction (upload → grade/reject, Grad-CAM tabs, simulation dashboard run
      + sweep), zero console errors. Docker + Hugging Face Spaces deployment
      config included (`Dockerfile`, `deploy/huggingface/`) — actually deploying
      needs the user's own HF account, so only the config and docs are provided.
- [~] Phase 6 — MATLAB compliance layer. **Partial, honestly split** — see
      `matlab/README.md` for the full breakdown. No MATLAB license was
      obtainable in this environment, so this phase could not be run against
      real MATLAB the way every other phase was run against real data. What
      actually is verified: `matlab/preprocess.m` (the retina crop+resize
      port) was run under GNU Octave — a free, MATLAB-syntax-compatible
      interpreter — against 20 real APTOS images and compared pixel-for-pixel
      against this project's own Python-cached output: mean abs diff
      1.489/255, p99 9.0/255 (gated, PASS), max 143/255 (a real, explained,
      thin-boundary-pixel effect from two different circle-fitting
      algorithms, not a bug — see `matlab/README.md`). Everything needing
      Deep Learning Toolbox or Simulink (`import_models.m`, `run_pipeline.m`,
      `evaluate.m`, `build_screening_workflow.m`) has no Octave equivalent to
      test against, so those are best-effort ports of the real Python logic,
      written but **not yet executed** — flagged as such rather than claimed.

## Messidor-2 test-set result (the actual PS headline number)

Run exactly once (`scripts/evaluate.py`; see `models/evaluate_run_log.json`),
on 1744 real, gradable Messidor-2 images (Google Brain's adjudicated grades —
see `data/splits/README.md` for the full label provenance), using the
threshold frozen from APTOS validation alone (0.395), never adjusted after
seeing this result:

| Metric | Result | 95% CI | PS target | Met? |
|---|---|---|---|---|
| Sensitivity | **0.2757** | [0.2355, 0.3140] | ≥ 0.90 | **No** |
| Specificity | 0.9736 | [0.9647, 0.9818] | ≥ 0.85 | Yes |
| AUC (DeLong) | 0.7544 | [0.7278, 0.7810] | — | — |
| QWK (5-class) | 0.3666 | — | — | — |

**This is a real, severe finding, not a bug: the sensitivity target is not
met on the actual mandated test set.** This is exactly the APTOS→Messidor-2
domain-shift risk AGENTS.md's risk table flagged as a required consideration
in advance ("expect and report a performance drop"), materializing more
severely than a mild drop. A read-only diagnostic (not used to pick a new
threshold — the frozen one above is the one, real, reported result) shows
why: on APTOS validation, referable cases score a median 0.958 (well above
the 0.395 threshold); on Messidor-2, referable cases score a median of only
0.053 — a systematic downward shift in the model's raw output scale between
the two datasets' images (almost certainly different camera/site
characteristics), not a loss of all signal — the AUC of 0.75 (vs ~0.97 on
APTOS validation) shows the model still ranks referable cases higher than
non-referable ones on average, just not at a magnitude the APTOS-derived
threshold can use.

**What this means honestly:** as built, this model is not ready to deploy
against Messidor-2-like images at the frozen threshold — a real clinical
deployment would need per-site recalibration (e.g. re-running Phase 4's
temperature scaling and threshold selection against a validation set drawn
from the deployment site's own camera/population) before trusting its
referable-DR calls, and this project has not done that recalibration or
validated it against a second independent test set. The specificity number
and the still-positive AUC are genuine, not spin, but they don't substitute
for the missed sensitivity target on the metric the problem statement
actually asks for.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Local training uses MPS on Apple Silicon (no CUDA). Heavy training runs on Kaggle
notebooks under `notebooks/` — see `AGENTS.md` for why.
