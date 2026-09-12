"""Parameter sweeps over the screening simulation, producing the three
findings for the slides:
  1. Reviewers needed per 100k patients, as a function of the auto-clear threshold
  2. Turnaround time vs bandwidth (2/5/10/20 Mbps)
  3. The centrepiece: auto-clear threshold vs reviewer load vs sensitivity lost

Writes Plotly HTML charts to results/simulation/ plus a CSV of raw results
per sweep. See AGENTS.md: that third curve is what judges should remember.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.drscreen.sim.screening_des import SimulationConfig, run_simulation

OUT_DIR = Path("results/simulation")
N_PATIENTS = 3000
SEED = 42


def sweep_reviewers_vs_threshold() -> pd.DataFrame:
    """For each auto-clear threshold, find the smallest reviewer count that
    keeps p95 turnaround under a 1-hour target -- "reviewers needed"."""
    rows = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        for n_reviewers in range(1, 21):
            config = SimulationConfig(n_patients=N_PATIENTS, seed=SEED, auto_clear_threshold=threshold, n_reviewers=n_reviewers)
            metrics = run_simulation(config)
            if metrics.turnaround_p95 <= 3600:
                rows.append({"threshold": threshold, "reviewers_needed": n_reviewers, "p95_turnaround_s": metrics.turnaround_p95})
                break
        else:
            rows.append({"threshold": threshold, "reviewers_needed": None, "p95_turnaround_s": metrics.turnaround_p95})
    return pd.DataFrame(rows)


def sweep_turnaround_vs_bandwidth() -> pd.DataFrame:
    rows = []
    for bandwidth in [2, 5, 10, 20]:
        config = SimulationConfig(n_patients=N_PATIENTS, seed=SEED, bandwidth_mbps=bandwidth)
        metrics = run_simulation(config)
        rows.append({"bandwidth_mbps": bandwidth, "p50_turnaround_s": metrics.turnaround_p50, "p95_turnaround_s": metrics.turnaround_p95})
    return pd.DataFrame(rows)


def sweep_threshold_tradeoff() -> pd.DataFrame:
    """The centrepiece: auto-clear threshold vs reviewer load vs sensitivity lost."""
    rows = []
    for threshold in [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        config = SimulationConfig(n_patients=N_PATIENTS, seed=SEED, auto_clear_threshold=threshold, n_reviewers=10)
        metrics = run_simulation(config)
        rows.append(
            {
                "threshold": threshold,
                "auto_cleared_frac": metrics.auto_cleared / metrics.patients_out,
                "reviewer_utilization": metrics.reviewer_utilization,
                "sensitivity_lost": metrics.sensitivity_lost,
            }
        )
    return pd.DataFrame(rows)


def _write_chart(fig: go.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs="cdn" instead of the default "inline" -- inline embeds
    # the entire Plotly.js bundle in every file (~4.3MB x 3 charts here for
    # a few KB of actual data); cdn loads it from a CDN at view time instead.
    fig.write_html(str(OUT_DIR / f"{name}.html"), include_plotlyjs="cdn")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reviewers_df = sweep_reviewers_vs_threshold()
    reviewers_df.to_csv(OUT_DIR / "reviewers_vs_threshold.csv", index=False)
    fig = go.Figure(go.Scatter(x=reviewers_df["threshold"], y=reviewers_df["reviewers_needed"], mode="lines+markers"))
    fig.update_layout(title="Reviewers needed vs auto-clear threshold (p95 turnaround < 1h)", xaxis_title="Auto-clear threshold", yaxis_title="Reviewers needed")
    _write_chart(fig, "reviewers_vs_threshold")

    bandwidth_df = sweep_turnaround_vs_bandwidth()
    bandwidth_df.to_csv(OUT_DIR / "turnaround_vs_bandwidth.csv", index=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bandwidth_df["bandwidth_mbps"], y=bandwidth_df["p50_turnaround_s"], mode="lines+markers", name="p50"))
    fig.add_trace(go.Scatter(x=bandwidth_df["bandwidth_mbps"], y=bandwidth_df["p95_turnaround_s"], mode="lines+markers", name="p95"))
    fig.update_layout(title="Turnaround time vs bandwidth", xaxis_title="Bandwidth (Mbps)", yaxis_title="Turnaround (s)")
    _write_chart(fig, "turnaround_vs_bandwidth")

    tradeoff_df = sweep_threshold_tradeoff()
    tradeoff_df.to_csv(OUT_DIR / "threshold_tradeoff.csv", index=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tradeoff_df["threshold"], y=tradeoff_df["reviewer_utilization"], mode="lines+markers", name="Reviewer utilization", yaxis="y1"))
    fig.add_trace(go.Scatter(x=tradeoff_df["threshold"], y=tradeoff_df["sensitivity_lost"], mode="lines+markers", name="Sensitivity lost", yaxis="y2"))
    fig.update_layout(
        title="Auto-clear threshold: reviewer load vs sensitivity lost (the centrepiece)",
        xaxis_title="Auto-clear threshold (P(grade=0) required to skip review)",
        yaxis=dict(title="Reviewer utilization", side="left"),
        yaxis2=dict(title="Sensitivity lost (referable cases missed)", overlaying="y", side="right"),
    )
    _write_chart(fig, "threshold_tradeoff")

    print(f"wrote 3 charts + CSVs to {OUT_DIR}")
    print(tradeoff_df.to_string(index=False))


if __name__ == "__main__":
    main()
