"""Generate a static, self-contained HTML rating form over 30 randomised
real cases -- 5-point Likert on heatmap plausibility, whether lesion
evidence supports the grade, and trust for triage. No server: the page
collects answers in-browser and offers a CSV download via a Blob link (see
AGENTS.md -- getting even one ophthalmology resident to fill this out is
the highest-value single action in this project).
"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.drscreen.pipeline import ScreeningPipeline
from src.drscreen.report.build import SEVERITY_LABELS, _overlay_lesions

N_CASES = 30
MANIFEST = Path("data/cache/aptos_val_manifest.csv")
OUT_PATH = Path("scripts/clinician_rating_form.html")


def _encode(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def build_cases(n: int, seed: int = 42) -> list[dict]:
    manifest = pd.read_csv(MANIFEST)
    sample = manifest.sample(n=min(n, len(manifest)), random_state=seed).reset_index(drop=True)

    pipeline = ScreeningPipeline()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sample))  # randomise display order, not just sample order

    cases = []
    for display_idx, i in enumerate(order):
        row = sample.iloc[int(i)]
        result = pipeline.run(row["path"])
        if result.quality_verdict == "REJECT":
            continue
        cases.append(
            {
                "case_id": f"case_{display_idx + 1:03d}",
                "original_uri": _encode(result.preprocessed_image),
                "lesion_overlay_uri": _encode(_overlay_lesions(result.preprocessed_image, result.lesion_masks)),
                "gradcam_uri": _encode(result.gradcam_overlay) if result.gradcam_overlay is not None else None,
                "severity_grade": result.severity_grade,
                "severity_label": SEVERITY_LABELS.get(result.severity_grade),
                "referable": result.referable,
            }
        )
    return cases


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Clinician Rating Harness</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; background: #F8FAFC; }}
  h1 {{ font-size: 20px; }}
  .case {{ background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; margin-bottom: 20px; }}
  .imgs {{ display: flex; gap: 8px; margin-bottom: 10px; }}
  .imgs img {{ width: 200px; border-radius: 6px; }}
  .meta {{ font-size: 13px; color: #475569; margin-bottom: 10px; }}
  .question {{ margin: 10px 0; }}
  .question label {{ font-size: 13px; font-weight: 600; display: block; margin-bottom: 4px; }}
  .likert {{ display: flex; gap: 14px; }}
  .likert label {{ font-weight: 400; display: flex; align-items: center; gap: 4px; }}
  #submit-bar {{ position: sticky; bottom: 0; background: #F8FAFC; padding: 12px 0; }}
  button {{ background: #3B82F6; color: white; border: none; border-radius: 6px; padding: 10px 18px; font-size: 14px; cursor: pointer; }}
  #status {{ margin-left: 12px; font-size: 13px; color: #475569; }}
</style></head>
<body>
<h1>DR Screening -- Clinician Rating Harness</h1>
<p>For each case, rate 1 (strongly disagree) to 5 (strongly agree). Ratings are not saved anywhere
except your own downloaded CSV -- nothing is sent to a server.</p>
<form id="rating-form">
{cases_html}
<div id="submit-bar"><button type="button" onclick="downloadCsv()">Download ratings as CSV</button><span id="status"></span></div>
</form>
<script>
const CASE_IDS = {case_ids_json};
function downloadCsv() {{
  const rows = [["case_id","heatmap_plausible","evidence_supports_grade","trust_for_triage","comment"]];
  let complete = true;
  for (const id of CASE_IDS) {{
    const get = (name) => {{
      const el = document.querySelector(`input[name="${{id}}_${{name}}"]:checked`);
      return el ? el.value : "";
    }};
    const heatmap = get("heatmap"), evidence = get("evidence"), trust = get("trust");
    const comment = (document.querySelector(`textarea[name="${{id}}_comment"]`) || {{value:""}}).value.replace(/"/g,'""');
    if (!heatmap || !evidence || !trust) complete = false;
    rows.push([id, heatmap, evidence, trust, `"${{comment}}"`]);
  }}
  const csv = rows.map(r => r.join(",")).join("\\n");
  const blob = new Blob([csv], {{type: "text/csv"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "clinician_ratings.csv"; a.click();
  document.getElementById("status").textContent = complete ? "Downloaded." : "Downloaded (some cases incomplete).";
}}
</script>
</body></html>
"""

_CASE_TEMPLATE = """
<div class="case">
  <div class="meta"><strong>{case_id}</strong> &middot; model grade: {severity_label} (grade {severity_grade}) &middot; referable: {referable}</div>
  <div class="imgs">
    <img src="{original_uri}" title="original">
    <img src="{lesion_overlay_uri}" title="lesion overlay">
    {gradcam_img}
  </div>
  <div class="question">
    <label>1. The Grad-CAM heatmap is clinically plausible for this grade.</label>
    <div class="likert">{heatmap_radios}</div>
  </div>
  <div class="question">
    <label>2. The lesion evidence (overlay + counts) supports the assigned grade.</label>
    <div class="likert">{evidence_radios}</div>
  </div>
  <div class="question">
    <label>3. I would trust this output for triage (deciding who needs an ophthalmologist sooner).</label>
    <div class="likert">{trust_radios}</div>
  </div>
  <div class="question">
    <label>Comment (optional)</label>
    <textarea name="{case_id}_comment" rows="2" style="width:100%;"></textarea>
  </div>
</div>
"""


def _likert(case_id: str, question: str) -> str:
    return "".join(
        f'<label><input type="radio" name="{case_id}_{question}" value="{v}">{v}</label>' for v in range(1, 6)
    )


def build_html(cases: list[dict]) -> str:
    cases_html = []
    for case in cases:
        gradcam_img = f'<img src="{case["gradcam_uri"]}" title="grad-cam">' if case["gradcam_uri"] else ""
        cases_html.append(
            _CASE_TEMPLATE.format(
                case_id=case["case_id"],
                severity_label=case["severity_label"],
                severity_grade=case["severity_grade"],
                referable=case["referable"],
                original_uri=case["original_uri"],
                lesion_overlay_uri=case["lesion_overlay_uri"],
                gradcam_img=gradcam_img,
                heatmap_radios=_likert(case["case_id"], "heatmap"),
                evidence_radios=_likert(case["case_id"], "evidence"),
                trust_radios=_likert(case["case_id"], "trust"),
            )
        )
    return _PAGE_TEMPLATE.format(cases_html="\n".join(cases_html), case_ids_json=json.dumps([c["case_id"] for c in cases]))


def main() -> None:
    cases = build_cases(N_CASES)
    html = build_html(cases)
    OUT_PATH.write_text(html)
    print(f"wrote {len(cases)} cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
