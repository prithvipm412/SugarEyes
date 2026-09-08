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
- [ ] Phase 1 — data spine, preprocessing, quality gate (in progress)
- [ ] Phase 2 — grading backbone and the headline sensitivity/specificity metric
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
