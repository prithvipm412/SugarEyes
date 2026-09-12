"""The PS-mandated ablation comparison: CNN-grading-only vs lesion-features-
only vs fused, with bootstrap CIs. Reported exactly as measured -- if fusion
doesn't win on AUC, that's the reported result (see AGENTS.md: never adjust
the ablation table to fit the narrative; if fusion loses, the calibration/
pointing-game columns are where it may still win)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..fusion.model import build_fusion_input, train_gbm_fusion, train_logistic_fusion
from .metrics import bootstrap_ci, roc_auc_delong_ci


def _auc_row(name: str, y_eval: np.ndarray, scores_eval: np.ndarray) -> dict:
    auc, lo, hi = roc_auc_delong_ci(y_eval, scores_eval)
    boot_point, boot_lo, boot_hi = bootstrap_ci(lambda yt, ys: roc_auc_delong_ci(yt, ys)[0], y_eval, scores_eval, n=500)
    return {"model": name, "auc": auc, "auc_ci_delong": (lo, hi), "auc_ci_bootstrap": (boot_lo, boot_hi)}


def run_ablation(
    grading_logits_train: np.ndarray,
    lesion_features_train: np.ndarray,
    y_train: np.ndarray,
    grading_logits_eval: np.ndarray,
    lesion_features_eval: np.ndarray,
    y_eval: np.ndarray,
    referable_score_eval: np.ndarray,
    seed: int = 42,
) -> list[dict]:
    """Three rows, each with real bootstrap + DeLong CIs:
      (a) CNN grading only -- the grading model's own referable_score, no fitting
      (b) Lesion features only -- logistic regression fit on lesion features alone
      (c) Fused -- both logistic and GBM fusion, both reported
    """
    rows = [_auc_row("CNN grading only", y_eval, referable_score_eval)]

    lesion_only = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)).fit(lesion_features_train, y_train)
    lesion_only_probs = lesion_only.predict_proba(lesion_features_eval)[:, 1]
    rows.append(_auc_row("Lesion features only", y_eval, lesion_only_probs))

    X_train = build_fusion_input(grading_logits_train, lesion_features_train)
    X_eval = build_fusion_input(grading_logits_eval, lesion_features_eval)
    for name, trainer in [("Fused (logistic)", train_logistic_fusion), ("Fused (GBM)", train_gbm_fusion)]:
        model = trainer(X_train, y_train, seed=seed)
        probs = model.predict_proba(X_eval)[:, 1]
        rows.append(_auc_row(name, y_eval, probs))

    return rows


def format_ablation_table(rows: list[dict]) -> str:
    lines = [f"{'model':<22} {'AUC':>8} {'95% CI (DeLong)':>20} {'95% CI (bootstrap)':>20}"]
    for row in rows:
        d_lo, d_hi = row["auc_ci_delong"]
        b_lo, b_hi = row["auc_ci_bootstrap"]
        lines.append(f"{row['model']:<22} {row['auc']:>8.4f} {f'[{d_lo:.4f}, {d_hi:.4f}]':>20} {f'[{b_lo:.4f}, {b_hi:.4f}]':>20}")
    return "\n".join(lines)


if __name__ == "__main__":
    import time

    t0 = time.time()
    rng = np.random.default_rng(23)
    n_train, n_eval = 300, 150

    def make_split(n):
        grading_logits = rng.normal(0, 1, (n, 4))
        lesion_features = rng.normal(0, 1, (n, 21))
        referable_score = 1 / (1 + np.exp(-grading_logits[:, 1]))
        y = (rng.random(n) < referable_score).astype(int)
        return grading_logits, lesion_features, y, referable_score

    gl_tr, lf_tr, y_tr, _rs_tr = make_split(n_train)
    gl_ev, lf_ev, y_ev, rs_ev = make_split(n_eval)

    rows = run_ablation(gl_tr, lf_tr, y_tr, gl_ev, lf_ev, y_ev, rs_ev)
    assert len(rows) == 4
    assert {r["model"] for r in rows} == {"CNN grading only", "Lesion features only", "Fused (logistic)", "Fused (GBM)"}
    for row in rows:
        assert 0.0 <= row["auc"] <= 1.0
        assert row["auc_ci_delong"][0] <= row["auc"] <= row["auc_ci_delong"][1]

    print(format_ablation_table(rows))
    elapsed = time.time() - t0
    print(f"ablation.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
