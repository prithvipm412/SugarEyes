"""Evaluation metrics for DR grading: sensitivity/specificity at a fixed
threshold, bootstrap CIs, quadratic weighted kappa, and AUC with a DeLong CI."""
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def sensitivity_specificity_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    """Sensitivity/specificity of the binary rule `y_score >= threshold`
    against binary ground truth `y_true` (e.g. referable-DR)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_score) >= threshold
    tp = np.sum(y_pred & y_true)
    fn = np.sum(~y_pred & y_true)
    tn = np.sum(~y_pred & ~y_true)
    fp = np.sum(y_pred & ~y_true)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return float(sensitivity), float(specificity)


def bootstrap_ci(metric_fn, y_true: np.ndarray, y_score: np.ndarray, n: int = 2000, confidence: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) for a scalar `metric_fn(y_true, y_score)`."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = float(metric_fn(y_true, y_score))

    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    boot_estimates = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        boot_estimates[i] = metric_fn(y_true[idx], y_score[idx])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot_estimates, [alpha, 1 - alpha])
    return point, float(lo), float(hi)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def _delong_components(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Structural components for DeLong's AUC variance (Sun & Xu, 2014)."""
    diff = pos_scores[:, None] - neg_scores[None, :]
    psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    return psi.mean(axis=1), psi.mean(axis=0)  # v10 (len m), v01 (len n)


def roc_auc_delong_ci(y_true: np.ndarray, y_score: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """AUC with a DeLong-method confidence interval (analytic, no resampling)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_scores, neg_scores = y_score[y_true == 1], y_score[y_true == 0]
    m, n = len(pos_scores), len(neg_scores)
    if m == 0 or n == 0:
        raise ValueError("roc_auc_delong_ci requires both positive and negative examples")

    v10, v01 = _delong_components(pos_scores, neg_scores)
    auc = v10.mean()
    s10 = v10.var(ddof=1) / m if m > 1 else 0.0
    s01 = v01.var(ddof=1) / n if n > 1 else 0.0
    se = np.sqrt(s10 + s01)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    lo, hi = auc - z * se, auc + z * se
    return float(auc), float(max(0.0, lo)), float(min(1.0, hi))


def confusion_matrix_5x5(y_true: np.ndarray, y_pred: np.ndarray, num_grades: int = 5) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_grades)))


def plot_confusion_matrix(cm: np.ndarray, out_path: str, class_names: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_names = class_names or [str(i) for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import tempfile
    import time
    from pathlib import Path

    t0 = time.time()
    rng = np.random.default_rng(5)

    # sensitivity/specificity: a perfectly separable toy case
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    sens, spec = sensitivity_specificity_at_threshold(y_true, y_score, threshold=0.5)
    assert sens == 1.0 and spec == 1.0

    # bootstrap_ci: point estimate matches direct computation, lo <= point <= hi
    y_true_n = rng.integers(0, 2, 100)
    y_score_n = rng.uniform(0, 1, 100)
    point, lo, hi = bootstrap_ci(lambda yt, ys: sensitivity_specificity_at_threshold(yt, ys, 0.5)[0], y_true_n, y_score_n, n=200)
    assert lo <= point <= hi

    # quadratic_weighted_kappa: perfect agreement -> kappa 1.0
    y_pred_perfect = np.array([0, 1, 2, 3, 4, 2, 1])
    assert quadratic_weighted_kappa(y_pred_perfect, y_pred_perfect) == 1.0

    # roc_auc_delong_ci: perfectly separable -> AUC 1.0, narrow CI
    auc, auc_lo, auc_hi = roc_auc_delong_ci(y_true, y_score)
    assert abs(auc - 1.0) < 1e-9
    assert auc_lo <= auc <= auc_hi

    # cross-check DeLong AUC against sklearn's AUC on a noisy case
    from sklearn.metrics import roc_auc_score

    y_true_noisy = rng.integers(0, 2, 300)
    y_score_noisy = y_true_noisy * 0.5 + rng.uniform(0, 1, 300)
    auc_delong, _, _ = roc_auc_delong_ci(y_true_noisy, y_score_noisy)
    auc_sklearn = roc_auc_score(y_true_noisy, y_score_noisy)
    assert abs(auc_delong - auc_sklearn) < 1e-9, (auc_delong, auc_sklearn)

    # confusion matrix + plot
    y_true_5 = rng.integers(0, 5, 50)
    y_pred_5 = rng.integers(0, 5, 50)
    cm = confusion_matrix_5x5(y_true_5, y_pred_5)
    assert cm.shape == (5, 5)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "cm.png"
        plot_confusion_matrix(cm, str(out_path))
        assert out_path.exists()

    elapsed = time.time() - t0
    print(f"metrics.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
