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
- [x] Phase 2 (locally) — CORN ordinal grading model, real gate pass on the same
      300-image sample: validation QWK 0.8036, threshold frozen at 0.49 (val
      sensitivity 0.9474 / specificity 0.9231), ONNX parity 1.2e-5. **Not yet the
      project's real result** — per `AGENTS.md`, heavy grading training is Kaggle-only
      on the full ~3662-image set (`notebooks/kaggle_train_grading.ipynb`), and the
      true headline sensitivity/specificity numbers need Messidor-2 (ADCIS
      registration — not available in this environment as of writing).
- [ ] Phase 3 — retinal structures and lesions
- [ ] Phase 4 — explainability, calibration, fusion, reports
- [ ] Phase 5 — screening simulation, API, and UI
- [ ] Phase 6 — MATLAB compliance layer

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Local training uses MPS on Apple Silicon (no CUDA). Heavy training runs on Kaggle
notebooks under `notebooks/` — see `AGENTS.md` for why.
