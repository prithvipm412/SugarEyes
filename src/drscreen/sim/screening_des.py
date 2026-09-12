"""District screening programme: a SimPy discrete-event model (replaces
Simulink/SimEvents -- see AGENTS.md; the MATLAB/Simulink equivalent is
Phase 6's screening_workflow.slx).

Models patient arrivals, image capture (nurse resource), upload (bandwidth-
limited), AI inference, and clinician review (an M/M/c-style queue) as a
network of SimPy resources. The auto-clear lever -- routing high-confidence
grade-0 cases straight through without review -- is the trade-off this
whole simulation exists to quantify.

The auto-clear decision and its "sensitivity lost" cost are driven by real
(true grade, P(grade==0)) pairs bootstrap-resampled from the real trained
grading model's actual validation-set behaviour (data/cache/fusion_dataset.npz,
built in Phase 4), not an assumed synthetic error rate -- this is what makes
the auto-clear-threshold-vs-reviewer-load-vs-sensitivity-lost curve a real
measurement grounded in this project's own model, not a generic queueing toy.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simpy

FUSION_DATA_PATH = Path("data/cache/fusion_dataset.npz")


def annual_volume_to_hourly_rate(patients_per_year: int, working_days_per_year: int = 250, working_hours_per_day: float = 8.0) -> float:
    """District-wide arrival rate implied by an annual patient volume, over
    working hours only (see AGENTS.md: "working-day distributed")."""
    return patients_per_year / (working_days_per_year * working_hours_per_day)


@dataclass
class SimulationConfig:
    n_patients: int = 1000
    n_centres: int = 20
    nurses_per_centre: int = 2
    capture_minutes_mean: float = 3.0
    bandwidth_mbps: float = 5.0
    image_size_mb: float = 5.0
    inference_seconds_per_image: float = 1.1  # measured: pipeline.py's real per-image latency
    n_reviewers: int = 5
    review_minutes_mean: float = 3.0
    auto_clear_threshold: float = 0.95  # P(grade==0) above which a case skips review
    # Default matches AGENTS.md's stated default scenario: 100,000 patients/year,
    # working-day distributed (250 days x 8h) -> 50/hour district-wide.
    arrivals_per_hour: float = field(default_factory=lambda: annual_volume_to_hourly_rate(100_000))
    seed: int = 42


@dataclass
class SimulationMetrics:
    patients_in: int = 0
    patients_out: int = 0
    auto_cleared: int = 0
    reviewed: int = 0
    referable_total: int = 0
    referable_auto_cleared: int = 0  # sensitivity lost: a referable case that skipped review
    turnaround_seconds: list = field(default_factory=list)
    review_queue_wait_seconds: list = field(default_factory=list)
    reviewer_busy_seconds: float = 0.0
    n_reviewers: int = 1
    total_sim_seconds: float = 0.0  # env.now at the end of the run -- the actual elapsed
    # simulated time, needed for utilization; NOT the same as any single patient's turnaround

    @property
    def sensitivity_lost(self) -> float:
        """Fraction of truly-referable patients who were auto-cleared
        without review -- undefined (not zero) if there were none."""
        if self.referable_total == 0:
            return float("nan")
        return self.referable_auto_cleared / self.referable_total

    @property
    def turnaround_p50(self) -> float:
        return float(np.median(self.turnaround_seconds)) if self.turnaround_seconds else float("nan")

    @property
    def turnaround_p95(self) -> float:
        return float(np.percentile(self.turnaround_seconds, 95)) if self.turnaround_seconds else float("nan")

    @property
    def reviewer_utilization(self) -> float:
        """Fraction of total reviewer-seconds (n_reviewers x elapsed time)
        actually spent reviewing. In [0, 1] by construction."""
        capacity_seconds = self.n_reviewers * self.total_sim_seconds
        return float("nan") if capacity_seconds == 0 else self.reviewer_busy_seconds / capacity_seconds


def _load_grade_confidence_pairs() -> tuple[np.ndarray, np.ndarray]:
    """Real (grade, P(grade==0)) pairs from the Phase 4 fusion dataset --
    what actually drives the simulated triage decision. Raises if missing
    rather than falling back to a synthetic distribution (see AGENTS.md)."""
    if not FUSION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{FUSION_DATA_PATH} missing -- run scripts/build_fusion_dataset.py first. "
            "The simulation's triage logic is grounded in the real trained model's "
            "behaviour, not a synthetic assumption, so this data is required."
        )
    data = np.load(FUSION_DATA_PATH)
    grades = data["grades"]
    p_grade_ge_1 = 1.0 / (1.0 + np.exp(-data["grading_logits"][:, 0]))  # sigmoid of the first CORN logit
    p_grade_eq_0 = 1.0 - p_grade_ge_1
    return grades, p_grade_eq_0


def run_simulation(config: SimulationConfig) -> SimulationMetrics:
    grades, p_grade_eq_0 = _load_grade_confidence_pairs()
    rng = np.random.default_rng(config.seed)
    metrics = SimulationMetrics(n_reviewers=max(1, config.n_reviewers))

    env = simpy.Environment()
    nurses = simpy.Resource(env, capacity=max(1, config.n_centres * config.nurses_per_centre))
    reviewers = simpy.Resource(env, capacity=max(1, config.n_reviewers))
    upload_seconds = (config.image_size_mb * 8) / config.bandwidth_mbps

    def patient_process(env, patient_idx: int):
        arrival_time = env.now
        metrics.patients_in += 1

        sample_idx = rng.integers(0, len(grades))
        true_grade = int(grades[sample_idx])
        confidence_grade0 = float(p_grade_eq_0[sample_idx])
        is_referable = true_grade >= 2
        if is_referable:
            metrics.referable_total += 1

        with nurses.request() as req:
            yield req
            yield env.timeout(max(0.0, rng.exponential(config.capture_minutes_mean * 60)))

        yield env.timeout(upload_seconds)
        yield env.timeout(config.inference_seconds_per_image)

        if confidence_grade0 >= config.auto_clear_threshold:
            metrics.auto_cleared += 1
            if is_referable:
                metrics.referable_auto_cleared += 1
        else:
            queue_start = env.now
            with reviewers.request() as req:
                yield req
                metrics.review_queue_wait_seconds.append(env.now - queue_start)
                review_seconds = max(0.0, rng.exponential(config.review_minutes_mean * 60))
                metrics.reviewer_busy_seconds += review_seconds
                yield env.timeout(review_seconds)
            metrics.reviewed += 1

        metrics.turnaround_seconds.append(env.now - arrival_time)
        metrics.patients_out += 1

    def arrival_process(env):
        mean_interarrival_seconds = 3600.0 / config.arrivals_per_hour
        for patient_idx in range(config.n_patients):
            yield env.timeout(rng.exponential(mean_interarrival_seconds))
            env.process(patient_process(env, patient_idx))

    env.process(arrival_process(env))
    env.run()  # no explicit `until` -- runs until every scheduled event (all
    # n_patients arrivals plus their downstream capture/review) completes,
    # which is exactly what makes patients_in == patients_out a real
    # invariant rather than an artifact of an arbitrary cutoff.
    metrics.total_sim_seconds = env.now

    return metrics


if __name__ == "__main__":
    import time

    t0 = time.time()
    if not FUSION_DATA_PATH.exists():
        print(f"SKIP: {FUSION_DATA_PATH} not found -- run scripts/build_fusion_dataset.py first")
    else:
        config = SimulationConfig(n_patients=5, seed=1)
        metrics = run_simulation(config)
        assert metrics.patients_in == 5
        assert metrics.patients_out == metrics.patients_in, "queue conservation violated"
        assert metrics.auto_cleared + metrics.reviewed == metrics.patients_out
        assert 0.0 <= (metrics.sensitivity_lost if not np.isnan(metrics.sensitivity_lost) else 0.0) <= 1.0

        elapsed = time.time() - t0
        print(
            f"screening_des.py smoke test ok in {elapsed:.3f}s -- "
            f"in={metrics.patients_in} out={metrics.patients_out} "
            f"auto_cleared={metrics.auto_cleared} reviewed={metrics.reviewed}"
        )
        assert elapsed < 30
