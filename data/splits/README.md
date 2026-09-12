# Data splits

Split manifests for this project are committed here as reproducibility artifacts (see
`AGENTS.md` — cached images and raw downloads are gitignored, splits are not).

Populated in Phase 1 (`scripts/make_splits.py`):
- `aptos_train.csv`, `aptos_val.csv` — stratified by grade, seed 42
- `messidor2_test.csv` — untouched until the single Phase 2 evaluation run

## Messidor-2 label provenance

ADCIS's own registration process could not be completed in this project's build
environment. Used instead: Google Brain's adjudicated re-grading of Messidor-2
(Kaggle mirror `google-brain/messidor2-dr-grades`; images from `xyaustin/messidor2`,
raw/original resolution, not any mirror's own preprocessed version), citation:

> Krause, J. et al. Grader variability and the importance of reference standards for
> evaluating machine learning models for diabetic retinopathy. Ophthalmology (2018).
> doi:10.1016/j.ophtha.2018.01.034

This is not a lower-quality stand-in for the original Messidor-2 grades — it is the
standard label source cited throughout the DR-screening ML literature specifically
*because* the original grades were noisier: three ophthalmologists graded each image
independently, with disagreements adjudicated by a retinal specialist.

**Grade mapping: identity, not a reconciliation.** `adjudicated_dr_grade` is
documented as "5 point ICDR grade" (0=None, 1=Mild, 2=Moderate, 3=Severe, 4=PDR) —
the same 0-4 ICDR scale APTOS and IDRiD already use here, confirmed by inspecting
`messidor_readme.txt` before writing any evaluation code, not assumed. No mapping
was applied because none was needed.

**Filtering**: of 1748 labeled images, 4 are marked `adjudicated_gradable == 0`
(no grade provided in the source data at all) and are excluded — see
`scripts/download_messidor2.py`. This mirrors what this project's own IQA quality
gate would do with an ungradable image (REJECT, not force a grade), and is decided
from a property of the images themselves, independent of and prior to running this
project's model on them.

Final test set: 1744 images. Grade distribution: 0=1017, 1=270, 2=347, 3=75, 4=35
(457 referable, grade >= 2).
