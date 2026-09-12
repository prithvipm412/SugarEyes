"""Temperature scaling for the referable-DR probability (grade >= 2), fit on
the VALIDATION set only (see AGENTS.md: test data is touched exactly once,
at the end). Operates on the binary referable probability directly, via a
logit-space temperature -- that's the actual calibrated quantity the report
and the referral decision use, not the 5-way grade distribution.
"""
import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-6


def _to_logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped))


def fit_temperature(raw_probs: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature T minimizing binary NLL of
    sigmoid(logit(raw_probs) / T) against binary `labels`."""
    logits = _to_logit(np.asarray(raw_probs))
    labels = np.asarray(labels, dtype=np.float64)

    def nll(t: float) -> float:
        scaled = 1.0 / (1.0 + np.exp(-logits / max(t, 1e-3)))
        scaled = np.clip(scaled, _EPS, 1 - _EPS)
        return float(-np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled)))

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def apply_temperature(raw_probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = _to_logit(np.asarray(raw_probs))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: weighted mean |accuracy - confidence| over confidence bins,
    where confidence is the probability assigned to the predicted class."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(np.float64)
    confidence = np.where(predictions == 1, probs, 1 - probs)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(accuracies[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, out_path: str, n_bins: int = 15, title: str = "Reliability diagram") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = np.asarray(probs)
    labels = np.asarray(labels)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(np.float64)
    confidence = np.where(predictions == 1, probs, 1 - probs)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, accs = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        centers.append((lo + hi) / 2)
        accs.append(accuracies[in_bin].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    if centers:
        ax.bar(centers, accs, width=1 / n_bins, alpha=0.7, edgecolor="black", label="observed")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import tempfile
    import time
    from pathlib import Path

    t0 = time.time()
    rng = np.random.default_rng(21)

    # Deliberately overconfident synthetic probs: correct ~70% of the time
    # but always reported near 0 or 1 -- a textbook miscalibration case.
    n = 500
    labels = rng.integers(0, 2, n)
    correct = rng.random(n) < 0.7
    raw_probs = np.where(labels == 1, np.where(correct, 0.95, 0.05), np.where(correct, 0.05, 0.95))
    raw_probs = np.clip(raw_probs + rng.normal(0, 0.02, n), 0.001, 0.999)

    ece_before = expected_calibration_error(raw_probs, labels)
    temperature = fit_temperature(raw_probs, labels)
    calibrated_probs = apply_temperature(raw_probs, temperature)
    ece_after = expected_calibration_error(calibrated_probs, labels)

    assert temperature > 1.0, f"expected T > 1 to soften overconfident probs, got {temperature}"
    assert ece_after < ece_before, f"ECE should improve: before={ece_before:.4f} after={ece_after:.4f}"
    print(f"ECE before={ece_before:.4f} after={ece_after:.4f} T={temperature:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "reliability.png"
        reliability_diagram(raw_probs, labels, str(out_path))
        assert out_path.exists()

    elapsed = time.time() - t0
    print(f"calibrate.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
