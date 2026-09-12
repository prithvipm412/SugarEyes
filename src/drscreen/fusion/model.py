"""Fusion model: combines the grading model's CORN logits with the Phase 3
structured lesion feature vector to predict referable DR. Both logistic
regression and gradient boosting are trained and compared -- report
whichever the numbers actually favor (see AGENTS.md: never adjust the
ablation table to fit the narrative)."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_fusion_input(grading_logits: np.ndarray, lesion_features: np.ndarray) -> np.ndarray:
    """Concatenate grading logits (N, 4) and lesion features (N, F) -> (N, 4+F)."""
    return np.concatenate([grading_logits, lesion_features], axis=1)


def train_logistic_fusion(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Standardized + logistic regression, as a single fitted pipeline.

    Grading logits and lesion feature counts/areas/distances live on very
    different scales -- fitting LogisticRegression on the raw concatenation
    doesn't converge within a reasonable iteration budget (verified: it
    silently produces a worse-than-necessary fit rather than erroring),
    which would understate fusion's real performance in the ablation table.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(X, y)
    return model


def train_gbm_fusion(X: np.ndarray, y: np.ndarray, seed: int = 42) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X, y)
    return model


if __name__ == "__main__":
    import time

    t0 = time.time()
    rng = np.random.default_rng(22)
    n = 200
    grading_logits = rng.normal(0, 1, (n, 4))
    lesion_features = rng.normal(0, 1, (n, 21))
    y = (grading_logits[:, 1] + lesion_features[:, 0] > 0).astype(int)

    X = build_fusion_input(grading_logits, lesion_features)
    assert X.shape == (n, 25)

    log_model = train_logistic_fusion(X, y)
    gbm_model = train_gbm_fusion(X, y)
    for model in (log_model, gbm_model):
        probs = model.predict_proba(X)[:, 1]
        assert probs.shape == (n,)
        assert ((probs >= 0) & (probs <= 1)).all()

    elapsed = time.time() - t0
    print(f"fusion/model.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
