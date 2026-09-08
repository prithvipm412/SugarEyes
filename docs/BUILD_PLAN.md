# Build plan — Explainable DR Screening (SIH 2026, PS 26038)

This is the phase-by-phase build script for this repo. Persistent project context (tech
stack, hard constraints, design system, engineering rules, known risks) lives in
`AGENTS.md` / `CLAUDE.md` at the repo root — both are read automatically by Claude Code /
Antigravity on every turn. This document holds the phase prompts and gates that get fed
in one at a time, plus the guardrails to paste when the agent drifts.

## How to use this document

1. `AGENTS.md` / `CLAUDE.md` are already in place at the repo root.
2. Feed **one phase prompt at a time** below. Do not paste the whole document as a single
   task — it produces shallow code across the board.
3. Every phase ends with a **gate command**. If the gate fails, fix it before moving on.
   Do not let the agent talk you past a failed gate.
4. Commit at the end of every phase with the tag the phase specifies.

## Build order summary

| Phase | Output | Gate |
|---|---|---|
| 1 | Cache, preprocessing, camera simulator, IQA model | IQA acc >= 0.80, mask integrity, < 1.5 s |
| 2 | Grading model, threshold, headline metrics | Sens >= 0.90, spec >= 0.85, ONNX parity |
| 3 | OD/fovea, vessels, 4-class lesion U-Net, MA cascade | Vessel Dice >= 0.75, MA candidate recall >= 0.85 |
| 4 | Grad-CAM, pointing metric, calibration, fusion, PDF | ECE drops, ablation complete, report < 8 s |
| 5 | SimPy model, sweeps, FastAPI, React UI | 1000-patient sim < 10 s, API < 8 s |
| 6 | MATLAB pipeline script, Simulink workflow | Grades match Python, sim within 5% |

Phase 2 is the go/no-go. Everything after it is presentation. Do not build Phase 3 while
Phase 2's gate is failing.

---

## Phase 1 — Data spine, preprocessing, quality gate

```
Build Phase 1: the data foundation and image-quality module.

DATASETS to wire up (download scripts in scripts/, all free):
- APTOS 2019 (Kaggle, 3662 images, grades 0-4) — via kaggle CLI, but ONLY on Kaggle notebooks.
  Locally, download just a 300-image sample for development.
- IDRiD (IEEE DataPort, 516 graded, 81 with pixel masks for MA/HE/EX/SE + optic disc)
- Messidor-2 (ADCIS, registration required) — TEST SET ONLY, write a loader but do not use it yet
- DRIVE + CHASE_DB1 + STARE (vessel ground truth, ~88 images total)
- EyeQ label CSVs (quality labels over EyePACS images — use on Kaggle only)

BUILD:

1. src/drscreen/preprocess/retina.py
   - detect_retina_circle(img): find the circular FOV via thresholding + largest contour
   - crop_to_retina(img): tight square crop, returns image + boolean FOV mask
   - resize_cached(img, size=448): Lanczos, preserve the mask
   - A single preprocess_image() that chains these and returns uint8 RGB + mask

2. src/drscreen/preprocess/enhance.py
   - clahe_green(img): CLAHE on the green channel and on L of LAB, both selectable
   - illumination_normalize(img): Ben Graham style — subtract a heavy Gaussian blur, rescale
   - denoise(img): bilateral or non-local means, benchmarked for speed
   - All functions must respect the FOV mask (never enhance the black surround)

3. src/drscreen/preprocess/degrade.py — the camera simulator
   - defocus_blur, motion_blur, uneven_illumination, haze, dust_specks, over/under_exposure
   - Each parameterised by severity 0-1
   - simulate_capture(img, severity, seed) composes a random realistic subset
   This serves as IQA augmentation AND as the demo input path (we have no fundus camera).

4. src/drscreen/quality/features.py — handcrafted quality metrics
   - laplacian_variance, tenengrad (focus)
   - illumination_uniformity (block-wise mean std across the FOV)
   - fov_coverage (fraction of frame that is retina)
   - vessel_visibility (Frangi response mean inside FOV) — a strong quality proxy
   - contrast, saturation_clipping

5. src/drscreen/quality/model.py + train.py
   - Small CNN (timm resnet18, 224px input) with the 6 handcrafted features concatenated
     into the classifier head
   - 3 classes: GRADEABLE / ENHANCEABLE / REJECT
   - Trainable locally on MPS. Batch size 16 max.
   - On REJECT, return a reason string derived from whichever handcrafted feature is worst
     ("out of focus", "underexposed", "partial field of view")

6. scripts/build_cache.py
   - Walks raw datasets, applies preprocess_image, writes 448x448 uint8 PNG to data/cache/
   - Parallel with multiprocessing, but cap workers at 4 (16 GB machine)
   - Writes a manifest CSV: path, dataset, split, grade, has_masks

7. scripts/make_splits.py
   - Stratified train/val split of APTOS by grade, seed 42
   - Messidor-2 is its own untouched test split
   - Writes to data/splits/*.csv — these get COMMITTED

GATE (must pass before Phase 2):
  python -m scripts.build_cache --sample 300 && \
  python -m src.drscreen.quality.train --config configs/iqa.yaml --epochs 5 && \
  python -m tests.gate_phase1

tests/gate_phase1.py must assert:
  - Cache round-trip: a cached image reloads to the same array
  - Every enhance function leaves the region outside the FOV mask untouched
  - IQA validation accuracy >= 0.80 on the held-out split
  - Full preprocess + IQA on one image completes in under 1.5 s on this machine

Commit as: phase1-data-and-quality
```

---

## Phase 2 — Grading backbone and the headline metric

```
Build Phase 2: DR severity grading. This phase decides whether the project is viable, so
build it before anything decorative.

1. src/drscreen/grading/ordinal.py
   - CORN ordinal loss for K=5 ordered classes (4 conditional sigmoid outputs)
   - decode_corn(logits) -> grade 0-4 via cumulative probabilities
   - referable_score(logits) -> a single scalar P(grade >= 2), which is what we threshold
   Write unit tests for the decode. This logic is easy to get subtly wrong.

2. src/drscreen/grading/model.py
   - timm backbone (default resnet34; make it config-swappable to resnet50/convnext_tiny)
   - Ordinal head with 4 outputs
   - forward() returns logits; expose the last conv block by name for Grad-CAM

3. src/drscreen/grading/train.py
   - Class-weighted sampling (APTOS is ~50% grade 0 — a naive model predicts all-zero)
   - Albumentations: rotate, flip, scale, colour jitter, PLUS degrade.simulate_capture at low
     severity as augmentation
   - Cosine LR, AdamW, mixed precision, early stopping on validation QWK
   - Checkpoint every epoch to a resumable path

4. notebooks/kaggle_train_grading.ipynb
   - Self-contained Kaggle notebook: attaches the APTOS dataset, clones this repo, trains,
     saves checkpoints to /kaggle/working, exports ONNX at opset 13
   - Must handle session restart by resuming from the last checkpoint

5. src/drscreen/eval/metrics.py
   - sensitivity_specificity_at_threshold
   - bootstrap_ci(metric_fn, y_true, y_score, n=2000) -> (point, lo, hi)
   - quadratic_weighted_kappa
   - roc_auc with DeLong CI
   - full 5x5 confusion matrix + plot

6. scripts/select_threshold.py
   - Sweeps the referable threshold on the VALIDATION set only
   - Picks the threshold that maximises specificity subject to sensitivity >= 0.90
   - Writes the chosen value to models/manifest.json and FREEZES it
   - Refuses to run if pointed at the test split — assert on the split name

7. scripts/evaluate.py
   - Loads the frozen threshold, runs once on Messidor-2, prints the full metric table with CIs
   - Prints a loud warning if invoked more than once (append to a run log)

CRITICAL — PARITY GATE:
Before trusting anything downstream, prove PyTorch and ONNX Runtime agree.
  scripts/check_parity.py: run 50 cached images through both, assert max abs logit
  difference < 1e-3. Preprocessing must be byte-identical on both paths.

GATE:
  python -m scripts.check_parity && python -m tests.gate_phase2

tests/gate_phase2.py must assert:
  - CORN decode is correct on hand-constructed logit vectors
  - Validation QWK >= 0.80
  - The threshold file exists and the test split has not been read
  - ONNX parity passes

Then, and only then, run scripts/evaluate.py once and record the result.
If sensitivity < 0.90 or specificity < 0.85, DO NOT proceed. Fix it here: bigger backbone,
more data (add an EyePACS subset on Kaggle), higher resolution, longer training.

Commit as: phase2-grading
```

---

## Phase 3 — Retinal structures and lesions

```
Build Phase 3: the clinical evidence layer that Phase 4 will cite.

1. src/drscreen/lesions/structures.py
   - locate_optic_disc(img, vessel_map): brightest-region candidate scoring combined with
     vessel convergence density. Return centre + radius.
   - locate_fovea(img, od_centre, vessel_map): ~2.5 disc diameters along the arcade axis,
     refined by darkest-region search in that neighbourhood
   - Both return a confidence score; both must degrade gracefully on poor images

2. src/drscreen/lesions/vessels.py
   - Lightweight U-Net (segmentation_models_pytorch, resnet18 encoder), trained on
     48x48 patches from DRIVE + CHASE_DB1 + STARE
   - Patch training turns ~88 images into ~200k samples — this is what makes the tiny
     dataset workable
   - vessel_metrics(mask): tortuosity, branching density, calibre — these feed the
     neovascularisation heuristic and the fusion model

3. src/drscreen/lesions/unet.py
   - ONE U-Net, 4 output channels: MA, haemorrhage, hard exudate, soft exudate
   - segmentation_models_pytorch Unet, resnet34 encoder, 384px input
   - Trained on IDRiD (81 images) with heavy augmentation and patch-based sampling
   - Combined Dice + BCE loss, per-channel weighting (soft exudates are rare)
   - Report per-channel Dice separately. Soft exudate performance will be poor. Say so.

4. src/drscreen/lesions/candidates.py — the microaneurysm cascade
   This is the "sub-pixel detection" the PS specifically asks for. Do not skip it.
   - Stage 1: candidate extraction. Inverted green channel -> morphological top-hat with
     a small structuring element -> subtract the vessel mask -> h-maxima -> connected
     components filtered by area and circularity. Should yield 50-500 candidates per image.
   - Stage 2: 48x48 patch classifier (small CNN, trainable locally on MPS) that scores
     each candidate as MA vs not
   - This is O(candidates), not O(pixels) — argue this explicitly in the README

5. src/drscreen/lesions/neovascularisation.py
   - Rule-based only: abnormal vessel tortuosity + branching density within 1 disc
     diameter of the optic disc, above a calibrated threshold
   - MUST return a flag that marks the output as rule-based, and the UI must display it
     differently from learned outputs

6. src/drscreen/lesions/features.py
   - Extract the structured feature vector the fusion model will consume:
     lesion counts per class, total area per class, mean distance to fovea per class,
     max lesion size, vessel tortuosity, NV flag, MA count in the macular region
   - This is the bridge between Phase 3 and Phase 4

7. notebooks/kaggle_train_lesions.ipynb — same pattern as Phase 2, exports ONNX opset 13

GATE:
  python -m tests.gate_phase3

Must assert:
  - OD localisation within 1 disc radius on >= 80% of IDRiD images with OD ground truth
  - Vessel Dice >= 0.75 on the DRIVE test split
  - MA candidate stage recall >= 0.85 on IDRiD MA ground truth BEFORE classification
    (recall at the candidate stage is the ceiling for the whole cascade — measure it)
  - Lesion U-Net per-channel Dice reported for all 4 classes, no channel silently skipped
  - features.py returns a fixed-length vector with no NaNs on 20 test images
  - ONNX parity for both new models

Commit as: phase3-lesions
```

---

## Phase 4 — Explainability, calibration, fusion, reports

```
Build Phase 4: the module this problem statement is named after.

1. src/drscreen/explain/gradcam.py
   - Use jacobgil/pytorch-grad-cam. GradCAM and GradCAM++ both, selectable.
   - Target the last conv block of the grading backbone
   - Return the raw CAM at native resolution plus an upsampled overlay
   - Explicitly record and expose the CAM's native grid size (e.g. 14x14) in the output —
     this honesty is part of the deliverable

2. src/drscreen/explain/pointing.py — the agreement metric
   DO NOT use IoU. A 14x14 upsampled CAM against 15-pixel microaneurysms scores near zero
   even when the model is behaving correctly.
   Implement instead:
   - pointing_game(cam, lesion_mask, top_k=0.10): fraction of connected lesion components
     whose centroid falls inside the top 10% of CAM activation
   - Report separately per lesion class. Expect good scores for exudates and haemorrhages,
     poor for microaneurysms. That difference IS the finding — write it up.

3. src/drscreen/explain/calibrate.py
   - Temperature scaling: one scalar, optimised with scipy on the VALIDATION set only
   - expected_calibration_error(probs, labels, n_bins=15)
   - reliability_diagram plot, before and after
   - The fitted temperature goes into models/manifest.json alongside the threshold

4. src/drscreen/fusion/model.py
   - Inputs: grading-model logits (4) + the Phase 3 lesion feature vector
   - Model: logistic regression AND gradient boosting, both, compared
   - Trained on validation-fold predictions, evaluated on test

5. src/drscreen/eval/ablation.py — the PS-mandated comparison
   Produce a single table with bootstrap CIs for:
     (a) CNN grading only
     (b) Lesion features only (classifier on hand-derived features)
     (c) Fused
   Report whatever the numbers actually are. If fusion does not win on AUC, report that and
   add calibration (ECE) and pointing-game columns where it may still win. Never adjust the
   table to fit the narrative.

6. src/drscreen/report/build.py + templates/report.html
   - Jinja2 template -> WeasyPrint -> one-page PDF
   - Layout, top to bottom: patient/session ID and timestamp; original image with lesion
     overlay; Grad-CAM panel; large severity grade with the colour code from AGENTS.md;
     calibrated confidence as a bar with the CI; lesion counts table; referral decision;
     a footer stating model version, dataset provenance, and that this is a screening aid
     requiring clinician confirmation
   - Design target: a clinician can reach the referral decision in under 10 seconds and
     verify the evidence in under 30

7. src/drscreen/pipeline.py — the orchestrator
   One function: run_pipeline(image_path) -> ScreeningResult dataclass containing every
   artifact above. Every module in the repo hangs off this.

8. scripts/clinician_rating_harness.py
   - Generates 30 randomised cases as a simple static HTML form
   - 5-point Likert on: is the heatmap clinically plausible; does the lesion evidence
     support the grade; would you trust this for triage
   - Writes results to CSV. If you can get even one ophthalmology resident to complete this,
     it is worth more than any other single hour spent on this project.

GATE:
  python -m tests.gate_phase4

Must assert:
  - Grad-CAM runs on the ONNX-imported model without error
  - Pointing-game scores computed per class on IDRiD, no class skipped
  - ECE strictly decreases after temperature scaling
  - The ablation table has all three rows with real CIs
  - A PDF report generates end-to-end from a raw image in under 8 s

Commit as: phase4-explainability
```

---

## Phase 5 — Screening simulation, API, and UI

```
Build Phase 5: the deployment story.

1. src/drscreen/sim/screening_des.py — SimPy discrete-event model
   Replaces Simulink/SimEvents. Model a district programme:
   - Patient arrivals: Poisson process, configurable rate. Default 100,000/year across
     N primary health centres, working-day distributed.
   - Stages as SimPy resources: image capture (nurse, N centres), upload (bandwidth-limited
     transfer time = filesize/bandwidth, configurable 2-20 Mbps), AI inference (throughput
     from your measured latency), clinician review (M/M/c queue, c = number of ophthalmologists)
   - Triage routing: cases above a confidence threshold on grade 0 auto-clear without review.
     This is the lever that matters.
   - Instrument: queue lengths, waiting times, reviewer utilisation, end-to-end turnaround
     p50/p95, cases auto-cleared

2. scripts/sweep_simulation.py
   Parameter sweeps that produce the findings for your slides:
   - Reviewers needed per 100k patients, as a function of the auto-clear threshold
   - Turnaround time vs bandwidth (2 / 5 / 10 / 20 Mbps)
   - The trade-off curve: auto-clear threshold vs reviewer load vs sensitivity lost
   That last curve is the result judges will remember. Make it the centrepiece.
   Output: Plotly charts saved as HTML, plus a CSV of raw results.

3. api/main.py — FastAPI
   - POST /screen (multipart image upload) -> full ScreeningResult as JSON + base64 overlays
   - GET /report/{session_id} -> the PDF
   - POST /simulate -> runs a SimPy scenario with supplied parameters, returns metrics
   - Load ONNX models once at startup, not per request
   - CORS for the local frontend

4. web/ — Vite + React + TypeScript + Tailwind + shadcn/ui + Framer Motion
   Apply the design system in AGENTS.md exactly.
   Screens:
   - Upload: drag-and-drop, plus a "simulate field capture" control that applies
     degrade.simulate_capture at a chosen severity. This is how we demo without a camera —
     make it an obvious, deliberate feature rather than hiding it.
   - Quality gate: animated verdict. On reject, show the reason and a recapture prompt.
   - Result: severity grade with the colour code, calibrated confidence bar with CI,
     Grad-CAM overlay with an opacity slider and a side-by-side toggle, lesion overlay with
     per-class visibility switches, lesion counts table
   - Evidence panel: pointing-game agreement score, with an inline note explaining the
     CAM resolution limit
   - Simulation dashboard: interactive sliders for arrival rate, bandwidth, reviewer count,
     auto-clear threshold; Recharts plots updating from the /simulate endpoint
   - Everything transitions with Framer Motion. Nothing pops.

5. Deployment
   - Dockerfile for the API
   - Hugging Face Spaces config (free tier) so you have a public demo URL
   - Precompute results for 6 showcase images and ship them as fixtures, so the live demo
     never waits on a cold pipeline

GATE:
  python -m tests.gate_phase5

Must assert:
  - A 1000-patient simulation completes in under 10 s
  - Queue conservation: patients in == patients out + patients still in system
  - API round-trip on a real image returns a complete result in under 8 s
  - Frontend builds clean with no TypeScript errors

Commit as: phase5-simulation-and-ui
```

---

## Phase 6 — MATLAB compliance layer

```
Build Phase 6: a thin MATLAB/Simulink layer for problem-statement compliance.

Context: this PS is MathWorks-sponsored and judged by MathWorks engineers. It asks for a
MATLAB-based pipeline and a Simulink workflow model. The Python build is the working system;
this layer proves the pipeline is reproducible in the required toolchain. Get free MATLAB
access through your university license or by emailing MathWorks. MATLAB Online works fine —
no local install needed.

Budget: two days, no more. Do not port the whole project.

1. matlab/import_models.m
   - net = importNetworkFromONNX("models/grading.onnx")
   - Same for the lesion U-Net
   - Save as .mat so import runs once
   - VERIFY FIRST, before anything else: check the import produces no placeholder or custom
     layers, and that gradCAM runs on the result. If gradCAM fails, the fallback is to
     rebuild resnet34 natively with imagePretrainedNetwork and fine-tune the head for a few
     epochs on CPU with trainnet — a few hours, and it gives you a genuinely
     MATLAB-trained network to show.

2. matlab/preprocess.m
   - Port retina.py and enhance.py: imcrop, imresize, adapthisteq, imgaussfilt
   - Must produce output matching the Python path. Write matlab/check_parity.m that loads
     10 cached Python outputs and asserts max abs difference below tolerance.

3. matlab/run_pipeline.m
   - Load image -> preprocess -> grade -> gradCAM -> overlay -> display
   - Under 80 lines. This is the script you show a judge who asks "is this MATLAB?"

4. matlab/evaluate.m
   - perfcurve for ROC, bootci for confidence intervals, confusionchart
   - Reproduce the Phase 2 headline numbers in MATLAB and confirm they match

5. matlab/screening_workflow.slx — Simulink model
   - Mirror the SimPy model structure
   - Use SimEvents if your license includes it (Entity Generator, Entity Queue, Entity Server,
     Entity Output Switch, Resource Pool). CHECK LICENSE AND APPLE SILICON AVAILABILITY FIRST
     at mathworks.com/support/requirements/apple-silicon.html
   - If SimEvents is unavailable: build it with Stateflow plus plain Simulink blocks, or drive
     a Simulink visualisation layer from MATLAB-side event logic. Either satisfies the PS.
   - Validate against SimPy: same parameters must produce statistically equivalent queue
     metrics. A cross-implementation agreement check is itself a strong slide.

6. matlab/README.md
   - State plainly which parts run in MATLAB, which in Python, and why (training compute).
   - Do not overclaim. Judges respect a clear architecture note far more than a vague one.

GATE:
  Run run_pipeline.m on 5 images and confirm grades match the Python pipeline exactly.
  Run the Simulink model and confirm queue metrics land within 5% of the SimPy result.

Commit as: phase6-matlab-compliance
```

---

## Appendix — Guardrails to paste when the agent drifts

Vibe-coding failure modes specific to this project. Paste the relevant line when you see it.

```
- You wrote a metric into a README/docstring/log without computing it. Remove it and replace
  with "NOT YET MEASURED" until real code produces the number.
- You added a try/except that swallows a failure and continues with defaults. Remove it. Fail loud.
- You used the test split for something other than the final single evaluation. Revert.
- You assumed CUDA. This machine has none. Use get_device().
- You used IoU for CAM/lesion agreement. Use the pointing-game metric — see AGENTS.md.
- You loaded the full dataset into memory. Stream it. There is 8 GB of headroom, total.
- You set batch size above 16 for local training. Reduce it.
- You started a training run with no checkpointing. Kaggle sessions die at 12 hours.
- You claimed the neovascularisation output is learned. It is rule-based. Label it.
- You adjusted the ablation table to make fusion look better. Report the measured result.
```
