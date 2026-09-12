# Phase 6 — MATLAB compliance layer

Problem statement 26038 is MathWorks-sponsored and judged by MathWorks
engineers; it asks for a MATLAB-based pipeline and a Simulink workflow model
alongside the working Python system. This directory is that layer.

**Read this before running anything here.** Unlike every other phase in this
project, the code in this directory has **not been run against real MATLAB**
— there is no MATLAB installation anywhere in the development environment
this was built in, and no free/university license was obtainable at build
time. Every other phase's gate is a real, measured pass on real data; this
phase's status is split, and the two halves should not be confused:

| File | Status |
|---|---|
| `preprocess.m` | **Verified.** Run under GNU Octave 11.3.0 (a free, open-source, MATLAB-syntax-compatible interpreter — see below) against 20 real APTOS images, compared pixel-for-pixel against this project's own Python-cached output. |
| `check_parity.m` | **Verified**, same run. Real measured result: mean abs diff 1.489/255, p99 9.0/255, max 143.0/255, 0.18% of pixels differ by more than 20 (out of 255). `PHASE 6 PREPROCESSING PARITY: PASS`. |
| `enhance.m` | Not verified. Octave's image package has no `adapthisteq`, so the CLAHE path couldn't be exercised even under Octave. Also not currently called from the live inference path in either language — see the file's own header. |
| `import_models.m`, `run_pipeline.m`, `evaluate.m` | Not verified. These need Deep Learning Toolbox (`importNetworkFromONNX`, `gradCAM`), which has no Octave equivalent at all. Written carefully from the real Python source (`src/drscreen/grading/model.py`, `ordinal.py`, `explain/gradcam.py`, `eval/metrics.py`) but never executed. |
| `build_screening_workflow.m` / `screening_workflow.slx` | Not verified, and the least likely of all of these to run unmodified — see its own header. A manual-build parameter table is given below as the more reliable path. |

If you get MATLAB access (see below), **run things in that order** and
report back what breaks — likely candidates for each file are called out in
that file's own header comment, so you're not debugging blind.

## Why GNU Octave, and what it does and doesn't prove

[GNU Octave](https://octave.org/) is a free, open-source numerical
computing environment designed to be syntax-compatible with MATLAB for the
core language and for several toolbox-equivalent packages (its `image`
package mirrors much of Image Processing Toolbox). Installed here via
`brew install octave` + `pkg install -forge image` — no license, no
account, no restricted access.

This makes it a genuinely useful stand-in for verifying `preprocess.m`:
every function it calls (`rgb2gray`, `bwconncomp`, `bwboundaries`,
`imresize` with a custom kernel, `padarray`) is a real, documented Image
Processing Toolbox function with the same name and signature in actual
MATLAB, so a clean run under Octave is real evidence this file is at least
syntactically correct and numerically sound — not a guess.

What it does **not** prove: Octave has no Deep Learning Toolbox, no
Simulink, no `gradCAM`, no `bootci`, no `perfcurve`, no `confusionchart`,
and no `adapthisteq`. None of the toolbox-specific files above could be
touched by this. "Verified under Octave" and "verified" are not the same
claim, and this document does not use the second one for anything Octave
couldn't actually exercise.

## The one real, measured result: preprocessing parity

`check_parity.m` compares this port's crop+resize against
`scripts/build_cache.py`'s real Python output (OpenCV) on the same raw
APTOS images, for every image where both `data/raw/...` and
`data/cache/...` exist locally. Two sources of expected (not buggy)
disagreement, both explained in `check_parity.m`'s header comment:

1. **FOV circle detection.** `crop_to_retina` fits a minimum enclosing
   circle to the thresholded retina blob. OpenCV's `minEnclosingCircle`
   and this port's Welzl's-algorithm implementation are different
   algorithms for the *same* well-defined geometric quantity, so they
   should agree closely — and empirically do — but boundary pixels right
   at the sub-pixel edge of that circle can land a pixel or two differently,
   which is a ~150-intensity-level jump right at that thin ring (bright
   retina against black background). This is the entire explanation for
   why max abs diff (143/255) is so much higher than p99 (9/255): it's a
   thin ring of pixels, not a systemic misalignment. Mean abs diff across
   whole images is 1.489/255.
2. **Resize kernel.** OpenCV's `INTER_LANCZOS4` is an 8-tap (a=4) Lanczos
   kernel; MATLAB/Octave's built-in `imresize` only ships `lanczos2`/
   `lanczos3`. `preprocess.m` passes a hand-written a=4 kernel via
   `imresize`'s documented custom-kernel form (`{@kernel_fn, size}`) to
   match exactly, with antialiasing turned off to match OpenCV's behaviour
   on downscale. This is a real, standard MATLAB `imresize` feature, not
   an Octave-only trick.

`check_parity.m`'s gate checks the 99th percentile, not the max, against a
tolerance of 25 (measured value: 9.0) — deliberately, because of point 1
above: a hard max-based gate would be measuring algorithmic disagreement at
a boundary neither implementation is "wrong" about, not a real correctness
problem. If you re-run this on a larger sample and the p99 drifts
meaningfully, that's the number to look at first.

## Manual Simulink build (recommended over `build_screening_workflow.m`)

`build_screening_workflow.m`'s header explains why: SimEvents block
library path strings and port-connection APIs are the single piece of this
whole phase most likely to be version-specific in a way that can't be
checked without MATLAB open. If it errors, build this by hand instead —
it should take under an hour, and every parameter below is taken directly
from `src/drscreen/sim/screening_des.py`'s real `SimulationConfig`
defaults, not invented for this table:

| Stage | SimEvents block | Parameter | Value |
|---|---|---|---|
| Arrivals | Time-Based Entity Generator | Exponential, mean interarrival | `3600 / 50` s (100k patients/yr, 250 days × 8h) |
| Capture | N-Server (capacity = nurses) | Capacity | 40 (20 centres × 2 nurses) |
| | | Service time | Exponential, mean 180s |
| Upload | Single Server | Service time | Constant, `5.0×8/5.0` = 8s (5MB image / 5Mbps) |
| AI inference | Single Server | Service time | Constant, 1.1s (measured pipeline.py latency) |
| Auto-clear routing | Entity Output Switch / Gate | Threshold on `confidence_grade0` attribute | 0.95 |
| Review | N-Server (capacity = reviewers) | Capacity | 5 |
| | | Service time | Exponential, mean 180s |

The `confidence_grade0` attribute per entity should be drawn from real
`(grade, P(grade==0))` pairs — export `data/cache/fusion_dataset.npz` to a
`.mat` (`scripts/`-style, not yet written) and sample from it, the same
way `screening_des.py`'s `_load_grade_confidence_pairs` does. Wiring a
placeholder/synthetic distribution instead would violate this project's
"never fabricate" rule (AGENTS.md) exactly the way `screening_des.py`'s own
header explains SimPy's version does not.

**Validate against SimPy**: run `python -m src.drscreen.sim.screening_des`
-style code with the same parameters and compare `reviewer_utilization` and
`sensitivity_lost` — BUILD_PLAN's own gate asks for agreement within 5%.
That comparison itself, if it lands close, is a stronger slide than either
simulation alone.

## Getting MATLAB access

No free/university license was available when this was built. If that
changes:

1. Check for a campus **Total Academic Headcount** license at
   [mathworks.com](https://www.mathworks.com) (sign in with a university
   email) — free desktop + MATLAB Online, common at Indian engineering
   colleges specifically.
2. Otherwise, MathWorks sells an individual **Student Suite** (MATLAB +
   Simulink + ~10 toolboxes including Deep Learning Toolbox) — see
   [mathworks.com/academia](https://www.mathworks.com/academia).
3. Since PS 26038 is MathWorks-sponsored, check with your SIH nodal centre
   or the SIH portal for a hackathon-specific temporary license before
   paying for anything.

## Running the verified part yourself

```bash
brew install octave
octave --no-gui --eval "pkg install -forge image"
octave --no-gui --eval "pkg load image; addpath('matlab'); check_parity(20)"
```

Everything else needs real MATLAB with Image Processing Toolbox, Deep
Learning Toolbox, Statistics and Machine Learning Toolbox, and (for the
Simulink half) Simulink + SimEvents:

```matlab
import_models          % writes models/grading_matlab.mat -- do this first
run_pipeline('data/cache/aptos_val/<some-file>.png')
evaluate                % reproduces Phase 2's headline numbers
build_screening_workflow  % or build screening_workflow.slx by hand, see above
```
