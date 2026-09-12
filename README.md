# SugarEyes — Explainable DR Screening

An open-source screening pipeline for diabetic retinopathy: given a colour fundus
photograph, it produces an image-quality verdict, an International Clinical DR
severity grade (0-4), a referable-DR decision with calibrated confidence, lesion
evidence (microaneurysms, haemorrhages, exudates, vessels, optic disc, fovea), a
Grad-CAM explanation with a quantified pointing-game agreement score, and a one-page
clinician-facing PDF report. It also includes a discrete-event simulation of a
district screening programme.

Built for Smart India Hackathon 2026, problem statement 26038.

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
      specificity 0.9480 — already past both PS targets), ONNX parity 1.7e-5.
      The official headline numbers still need Messidor-2 (ADCIS registration —
      not available in this environment as of writing).
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

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Local training uses MPS on Apple Silicon (no CUDA). Heavy training runs on Kaggle
notebooks under `notebooks/` — see `AGENTS.md` for why.
