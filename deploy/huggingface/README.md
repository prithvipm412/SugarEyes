---
title: SugarEyes DR Screening
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# SugarEyes -- Explainable DR Screening (demo API)

This Space runs the SugarEyes screening API (see the repo root's `AGENTS.md` /
`README.md` for the full project). It serves `POST /screen`, `GET /report/{id}`,
and `POST /simulate` -- the same FastAPI app as `api/main.py`, built from the
repo-root `Dockerfile`.

**Model weights are not baked into this Space's image** (see `Dockerfile`'s
comment) -- they're expected to be present at `models/{grading,vessels,
lesion_unet,ma_classifier}/best.pt` and `models/iqa/best.pt` at container
start, via a Space-level persistent storage mount or a startup fetch step.
Configure that before expecting `/screen` to return real results; `/health`
will report `models_loaded: false` until it can.

This is a screening aid demo, not a certified diagnostic device -- see the
disclaimer on every generated PDF report.
