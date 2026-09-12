"""Phase 5 gate: 1000-patient simulation timing, queue conservation, API
round-trip timing, and a clean frontend build. See docs/BUILD_PLAN.md."""
import subprocess
import time
from pathlib import Path


def check_simulation_performance() -> None:
    from src.drscreen.sim.screening_des import SimulationConfig, run_simulation

    t0 = time.perf_counter()
    metrics = run_simulation(SimulationConfig(n_patients=1000, seed=42))
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"1000-patient simulation took {elapsed:.2f}s >= 10s"
    print(f"[ok] 1000-patient simulation in {elapsed:.3f}s")

    # Queue conservation: since run_simulation runs every scheduled event to
    # completion (see screening_des.py's own comment on this), patients
    # still in the system at the end must be exactly zero.
    patients_still_in_system = metrics.patients_in - metrics.patients_out
    assert patients_still_in_system == 0, f"queue conservation violated: {patients_still_in_system} patients unaccounted for"
    assert metrics.auto_cleared + metrics.reviewed == metrics.patients_out
    print(f"[ok] queue conservation: in={metrics.patients_in} == out={metrics.patients_out} + in_system=0")


def check_api_roundtrip() -> None:
    import pandas as pd
    from fastapi.testclient import TestClient

    from api.main import app

    manifest_path = Path("data/cache/aptos_val_manifest.csv")
    assert manifest_path.exists(), f"{manifest_path} missing -- run Phase 2's cache_aptos_val first"
    manifest = pd.read_csv(manifest_path)
    row = manifest.iloc[0]

    with TestClient(app) as client:  # triggers startup -> loads models once, not part of the timed budget
        t0 = time.perf_counter()
        with open(row["path"], "rb") as f:
            response = client.post("/screen", files={"file": ("image.png", f, "image/png")})
        elapsed = time.perf_counter() - t0

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["quality_verdict"] in ("GRADEABLE", "ENHANCEABLE", "REJECT")
    assert elapsed < 8.0, f"API round-trip took {elapsed:.2f}s >= 8s"
    print(f"[ok] API round-trip in {elapsed:.3f}s (verdict={body['quality_verdict']})")


def check_frontend_builds_clean() -> None:
    web_dir = Path("web")
    assert (web_dir / "package.json").exists(), f"{web_dir} missing -- Phase 5's frontend was not scaffolded"
    result = subprocess.run(["npm", "run", "build"], cwd=web_dir, capture_output=True, text=True)
    assert result.returncode == 0, f"frontend build failed:\n{result.stdout}\n{result.stderr}"
    assert "error TS" not in result.stdout, f"TypeScript errors in build output:\n{result.stdout}"
    print("[ok] frontend builds clean with no TypeScript errors")


def main() -> None:
    check_simulation_performance()
    check_api_roundtrip()
    check_frontend_builds_clean()
    print("PHASE 5 GATE: PASS")


if __name__ == "__main__":
    main()
