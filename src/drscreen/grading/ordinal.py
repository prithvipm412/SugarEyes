"""CORN ordinal loss and decoding (Shi, Cao & Raschka, 2022) for K ordered
DR severity grades (0-4), using K-1 conditional binary logits.

Each task k in [0, K-2] represents the conditional probability
P(grade > k | grade >= k). Training only includes examples with
grade >= k for task k (conditional subsetting) -- this is what makes CORN
rank-consistent, unlike naive independent binary classifiers per threshold.
"""
import torch
import torch.nn.functional as F

NUM_GRADES = 5  # ICDR 0-4


def corn_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CORN loss. `logits` is [N, NUM_GRADES-1], `labels` is [N] in [0, NUM_GRADES-1]."""
    num_tasks = logits.shape[1]
    total, count = logits.new_zeros(()), 0
    for k in range(num_tasks):
        mask = labels >= k
        if mask.sum() == 0:
            continue
        task_logits = logits[mask, k]
        task_targets = (labels[mask] > k).float()
        total = total + F.binary_cross_entropy_with_logits(task_logits, task_targets, reduction="sum")
        count += int(mask.sum().item())
    return total / max(count, 1)


def _cumulative_probs(logits: torch.Tensor) -> torch.Tensor:
    """P(grade > k) for k = 0..NUM_GRADES-2, via the CORN product rule:
    P(grade > k) = P(grade > k | grade >= k) * P(grade > k-1)."""
    conditional_probs = torch.sigmoid(logits)
    return torch.cumprod(conditional_probs, dim=1)


def decode_corn(logits: torch.Tensor) -> torch.Tensor:
    """Decode CORN logits to an integer grade in [0, NUM_GRADES-1].

    Rank-consistent by construction: `_cumulative_probs` is monotonically
    non-increasing along dim 1, so thresholding at 0.5 always yields a
    prefix of Trues -- the count of which is the decoded grade.
    """
    cumulative_probs = _cumulative_probs(logits)
    return (cumulative_probs > 0.5).sum(dim=1)


def referable_score(logits: torch.Tensor) -> torch.Tensor:
    """P(grade >= 2), i.e. P(grade > 1) -- the scalar thresholded for the
    referable-DR decision. Index 1 of the cumulative probs (0-indexed:
    index k holds P(grade > k))."""
    cumulative_probs = _cumulative_probs(logits)
    return cumulative_probs[:, 1]


if __name__ == "__main__":
    import time

    t0 = time.time()

    # decode_corn on hand-constructed logit vectors (large magnitude so
    # sigmoid saturates near 0 or 1, making the expected decode unambiguous).
    BIG = 10.0
    cases = [
        ([-BIG, -BIG, -BIG, -BIG], 0),
        ([BIG, -BIG, -BIG, -BIG], 1),
        ([BIG, BIG, -BIG, -BIG], 2),
        ([BIG, BIG, BIG, -BIG], 3),
        ([BIG, BIG, BIG, BIG], 4),
    ]
    for logit_vals, expected in cases:
        logits = torch.tensor([logit_vals])
        decoded = decode_corn(logits).item()
        assert decoded == expected, f"decode_corn({logit_vals}) = {decoded}, expected {expected}"

    # A non-saturated, hand-computed case: p = [0.9, 0.9, 0.1, 0.9]
    # cumprod = [0.9, 0.81, 0.081, 0.0729] -> thresholded [T, T, F, F] -> grade 2
    logit = lambda p: torch.logit(torch.tensor(p))
    logits = torch.stack([logit(0.9), logit(0.9), logit(0.1), logit(0.9)]).unsqueeze(0)
    assert decode_corn(logits).item() == 2

    # referable_score = P(grade > 1) = cumprod[1] = 0.9*0.9 = 0.81
    score = referable_score(logits).item()
    assert abs(score - 0.81) < 1e-4, score

    # rank consistency: cumulative probs must be non-increasing for random logits
    rng_logits = torch.randn(200, NUM_GRADES - 1)
    cum = _cumulative_probs(rng_logits)
    diffs = cum[:, 1:] - cum[:, :-1]
    assert (diffs <= 1e-6).all(), "cumulative probs must be non-increasing"

    # corn_loss runs and is finite/nonnegative on a random batch with all grades present
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4])
    logits = torch.randn(8, NUM_GRADES - 1, requires_grad=True)
    loss = corn_loss(logits, labels)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    assert logits.grad is not None

    elapsed = time.time() - t0
    print(f"ordinal.py smoke test ok in {elapsed:.3f}s")
    assert elapsed < 30
