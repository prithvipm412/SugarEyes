"""FastAPI serving layer: POST /screen for a single-image screening result,
GET /report/{session_id} for the PDF, POST /simulate for a SimPy scenario.
Models load once at startup (see AGENTS.md), not per request.

Session storage is a simple in-memory dict -- adequate for this project's
scope (a demo deliverable, not a production system with a database or
multiple worker processes). It does not persist across restarts and won't
work correctly behind more than one worker process; a real deployment
would replace this with a shared store.
"""
import base64
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.drscreen.pipeline import ScreeningPipeline, ScreeningResult
from src.drscreen.preprocess.degrade import simulate_capture
from src.drscreen.report.build import overlay_lesions, render_report_pdf
from src.drscreen.sim.screening_des import SimulationConfig, run_simulation

_pipeline: ScreeningPipeline | None = None
_sessions: dict[str, ScreeningResult] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pipeline
    _pipeline = ScreeningPipeline()
    yield


app = FastAPI(title="SugarEyes Screening API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode(img: np.ndarray | None) -> str | None:
    if img is None:
        return None
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("failed to encode image")
    return base64.b64encode(buf).decode("ascii")


def _sanitize_nan(value):
    """Recursively replace float('nan') with None. Python's json module
    emits a literal `NaN` token for it, which is invalid per the JSON spec
    -- browsers' JSON.parse throws on it, so anything that can genuinely be
    NaN (pointing-game score with no lesions, sensitivity_lost with no
    referable cases) must be sanitized before it reaches a JSON response."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_nan(v) for v in value]
    return value


def _serialize(result: ScreeningResult, session_id: str) -> dict:
    lesion_overlay = None
    if result.lesion_masks and result.preprocessed_image is not None:
        lesion_overlay = _encode(overlay_lesions(result.preprocessed_image, result.lesion_masks))

    return {
        "session_id": session_id,
        "quality_verdict": result.quality_verdict,
        "quality_reject_reason": result.quality_reject_reason,
        "severity_grade": result.severity_grade,
        "referable": result.referable,
        "raw_referable_score": result.raw_referable_score,
        "calibrated_confidence": result.calibrated_confidence,
        "lesion_counts": result.lesion_counts,
        "od_center": result.od_center,
        "od_confidence": result.od_confidence,
        "fovea_center": result.fovea_center,
        "fovea_confidence": result.fovea_confidence,
        "neovascularisation": result.neovascularisation,
        "gradcam_native_size": result.gradcam_native_size,
        "pointing_game_scores": result.pointing_game_scores,
        "images": {
            "original": _encode(result.preprocessed_image),
            "lesion_overlay": lesion_overlay,
            "gradcam": _encode(result.gradcam_overlay),
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": _pipeline is not None}


@app.post("/screen")
async def screen(file: UploadFile = File(...), degrade_severity: float = Form(0.0)) -> dict:
    """`degrade_severity` (0-1): applies the real camera-degradation
    simulator (see AGENTS.md -- this project has no physical fundus camera,
    so this is the deliberate, honestly-labeled way to demo field
    conditions) to the uploaded image before running the pipeline. 0 means
    the image is used as uploaded."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="models not loaded yet")

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if degrade_severity > 0:
        image = simulate_capture(image, severity=min(degrade_severity, 1.0), seed=uuid.uuid4().int % (2**31))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name

    try:
        result = _pipeline.run(tmp_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"could not process image: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = result
    return _sanitize_nan(_serialize(result, session_id))


@app.get("/report/{session_id}")
def get_report(session_id: str) -> FileResponse:
    result = _sessions.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="session not found")

    out_path = Path(tempfile.gettempdir()) / f"report_{session_id}.pdf"
    render_report_pdf(result, str(out_path), session_id=session_id, model_version="phase5-api")
    return FileResponse(str(out_path), media_type="application/pdf", filename=f"screening_report_{session_id}.pdf")


class SimulateRequest(BaseModel):
    n_patients: int = 1000
    n_centres: int = 20
    nurses_per_centre: int = 2
    bandwidth_mbps: float = 5.0
    n_reviewers: int = 5
    auto_clear_threshold: float = 0.95
    arrivals_per_hour: float | None = None
    seed: int = 42


@app.post("/simulate")
def simulate(req: SimulateRequest) -> dict:
    kwargs = req.model_dump(exclude={"arrivals_per_hour"})
    if req.arrivals_per_hour is not None:
        kwargs["arrivals_per_hour"] = req.arrivals_per_hour
    config = SimulationConfig(**kwargs)

    try:
        metrics = run_simulation(config)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return _sanitize_nan(
        {
            "patients_in": metrics.patients_in,
            "patients_out": metrics.patients_out,
            "auto_cleared": metrics.auto_cleared,
            "reviewed": metrics.reviewed,
            "sensitivity_lost": metrics.sensitivity_lost,
            "turnaround_p50_s": metrics.turnaround_p50,
            "turnaround_p95_s": metrics.turnaround_p95,
            "reviewer_utilization": metrics.reviewer_utilization,
        }
    )


if __name__ == "__main__":
    import time

    import pandas as pd
    from fastapi.testclient import TestClient

    manifest_path = Path("data/cache/aptos_val_manifest.csv")
    if not manifest_path.exists():
        print("SKIP: data/cache/aptos_val_manifest.csv not found -- run Phase 2's cache_aptos_val first")
    else:
        t0 = time.time()
        with TestClient(app) as client:  # triggers the startup event -> loads models once
            health = client.get("/health").json()
            assert health["models_loaded"] is True

            manifest = pd.read_csv(manifest_path)
            row = manifest.iloc[0]
            with open(row["path"], "rb") as f:
                response = client.post("/screen", files={"file": ("image.png", f, "image/png")})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["quality_verdict"] in ("GRADEABLE", "ENHANCEABLE", "REJECT")
            session_id = body["session_id"]

            # degrade_severity should be able to push a gradeable image to REJECT
            with open(row["path"], "rb") as f:
                degraded_response = client.post(
                    "/screen", files={"file": ("image.png", f, "image/png")}, data={"degrade_severity": "0.9"}
                )
            assert degraded_response.status_code == 200, degraded_response.text

            if body["quality_verdict"] != "REJECT":
                assert body["images"]["original"] is not None
                report_response = client.get(f"/report/{session_id}")
                assert report_response.status_code == 200
                assert report_response.headers["content-type"] == "application/pdf"

            sim_response = client.post("/simulate", json={"n_patients": 200})
            assert sim_response.status_code == 200, sim_response.text
            sim_body = sim_response.json()
            assert sim_body["patients_in"] == sim_body["patients_out"] == 200

        elapsed = time.time() - t0
        print(f"api/main.py smoke test ok in {elapsed:.3f}s")
        assert elapsed < 30
