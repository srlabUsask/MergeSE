"""Unit tests for the core merge math.

These exercise the algorithmic kernels without needing any HuggingFace
checkpoints - they construct synthetic state dicts directly.
"""
import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mergese import (  # noqa: E402
    _trim_by_percentile,
    _dare_drop,
    _elect_sign,
    _merge_with_signs,
    ties_merge,
    dare_ties_merge,
    average_merge,
    pcb_merge,
    _pcb_scores,
    _pcb_threshold,
    _minmax_normalize,
    _cosine_similarity,
    _sign_agreement,
)


def _sd(seed: int, shape=(8, 8)):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(*shape, generator=g)}


def test_trim_zeros_below_percentile():
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    trimmed, frac = _trim_by_percentile({"w": t}, 20.0)
    assert (trimmed["w"] == 0).sum().item() == 2  # bottom 20%
    assert math.isclose(frac, 0.2, rel_tol=1e-6)


def test_trim_handles_huge_tensors():
    """Regression: torch.quantile() fails for >16M-element tensors; the embedding
    table of a RoBERTa-base checkpoint is 50265 × 768 ≈ 38.6M elements."""
    big = torch.randn(50265, 768)
    trimmed, frac = _trim_by_percentile({"w": big}, 20.0)
    assert trimmed["w"].shape == big.shape
    assert 0.18 < frac < 0.22  # ~20% trimmed, allowing for ties at the cutoff


def test_dare_drops_and_rescales():
    t = torch.ones(10000)
    gen = torch.Generator().manual_seed(0)
    out = _dare_drop({"w": t}, 0.3, gen)["w"]
    # Mean should be ≈ 1.0 because non-dropped entries are scaled by 1/(1-p)
    assert abs(out.mean().item() - 1.0) < 0.05
    # ~30% of entries should be zero
    zero_frac = (out == 0).float().mean().item()
    assert 0.25 < zero_frac < 0.35


def test_elect_sign_majority():
    d1 = {"w": torch.tensor([+1.0, -1.0, +1.0])}
    d2 = {"w": torch.tensor([+1.0, +1.0, -1.0])}
    d3 = {"w": torch.tensor([-1.0, +1.0, +1.0])}
    elected = _elect_sign([d1, d2, d3], [1, 1, 1])["w"]
    # +,+,+
    assert torch.equal(elected, torch.tensor([1.0, 1.0, 1.0]))


def test_average_merge_reduces_to_mean_delta():
    base = {"w": torch.zeros(4)}
    deltas = [
        {"w": torch.tensor([1.0, 1.0, 1.0, 1.0])},
        {"w": torch.tensor([3.0, 3.0, 3.0, 3.0])},
    ]
    merged, stats = average_merge(base, deltas, [1.0, 1.0])
    assert torch.allclose(merged["w"], torch.tensor([2.0, 2.0, 2.0, 2.0]))


def test_ties_merge_resolves_conflicts():
    base = {"w": torch.zeros(4)}
    deltas = [
        {"w": torch.tensor([+1.0, +1.0, -1.0,  0.0])},
        {"w": torch.tensor([+1.0, -1.0, -1.0, +0.5])},
    ]
    merged, stats = ties_merge(base, deltas, [1.0, 1.0], 0.0)
    # Position 0: both +; should be ≈ +1
    assert merged["w"][0].item() > 0.5
    # Position 2: both -; should be ≈ -1
    assert merged["w"][2].item() < -0.5
    assert stats["method"] == "ties"


def test_dare_ties_runs_and_is_deterministic():
    base = {"w": torch.zeros(64)}
    g = torch.Generator().manual_seed(0)
    d1 = {"w": torch.randn(64, generator=g)}
    d2 = {"w": torch.randn(64, generator=g)}
    a, _ = dare_ties_merge(base, [d1, d2], [1.0, 1.0], 20.0, 0.3, seed=7)
    b, _ = dare_ties_merge(base, [d1, d2], [1.0, 1.0], 20.0, 0.3, seed=7)
    assert torch.allclose(a["w"], b["w"])


def test_minmax_normalize_spans_unit_interval():
    x = torch.tensor([[1.0, 3.0, 5.0], [-2.0, 0.0, 2.0]])
    y = _minmax_normalize(x, dim=1)
    assert torch.allclose(y.amin(dim=1), torch.zeros(2))
    assert torch.allclose(y.amax(dim=1), torch.ones(2))
    # A constant row must not divide by zero.
    z = _minmax_normalize(torch.ones(1, 4), dim=1)
    assert torch.isfinite(z).all()


def test_pcb_scores_penalise_cross_task_conflict():
    """Position 1 has a 2-vs-1 disagreement; every other position is unanimous.

    All magnitudes are ±1, so the intra-balancing term is flat and any score
    difference comes purely from inter-balancing (cross-task competition).
    """
    d1 = torch.tensor([1.0, +1.0, 1.0, 1.0])
    d2 = torch.tensor([1.0, +1.0, 1.0, 1.0])
    d3 = torch.tensor([1.0, -1.0, 1.0, 1.0])
    s = _pcb_scores([d1, d2, d3])
    assert s.shape == (3, 4)
    # Unanimous position scores positive for every task.
    assert (s[:, 0] > 0).all()
    # At the contested position the majority still scores positive...
    assert s[0, 1] > 0 and s[1, 1] > 0
    # ...while the dissenting task is pushed negative, so it loses the drop.
    assert s[2, 1] < 0
    # Consensus beats contested for the majority tasks.
    assert s[0, 0] > s[0, 1]


def test_pcb_scores_zero_out_a_deadlocked_position():
    """Two tasks in exact opposition cancel: no consensus, so no update."""
    d1 = torch.tensor([1.0, +1.0, 1.0, 1.0])
    d2 = torch.tensor([1.0, -1.0, 1.0, 1.0])
    s = _pcb_scores([d1, d2])
    assert torch.allclose(s[:, 1], torch.zeros(2), atol=1e-6)
    assert (s[:, 0] > 0).all()


def test_pcb_scores_favour_a_tasks_own_dominant_parameters():
    """Intra-balancing: within one task, the large update outranks the noise."""
    d1 = torch.tensor([5.0, 0.01, 0.01, 0.01])
    d2 = torch.tensor([5.0, 0.01, 0.01, 0.01])
    s = _pcb_scores([d1, d2])
    assert s[0, 0] > s[0, 1]


def test_pcb_threshold_keeps_requested_fraction():
    scores = torch.arange(1000, dtype=torch.float32)
    thr = _pcb_threshold(scores, 0.1)
    kept = (scores > thr).sum().item()
    assert 95 <= kept <= 105
    # ratio of 1.0 means "keep everything" -> no threshold at all
    assert _pcb_threshold(scores, 1.0) is None


def test_pcb_merge_prefers_the_consensus_direction():
    base = {"w": torch.zeros(4)}
    deltas = [
        {"w": torch.tensor([+1.0, +1.0, +1.0, +1.0])},
        {"w": torch.tensor([+1.0, +1.0, +1.0, -1.0])},
        {"w": torch.tensor([+1.0, +1.0, +1.0, -1.0])},
    ]
    merged, stats = pcb_merge(base, deltas, [1.0, 1.0, 1.0], ratio=1.0, lam=1.0)
    assert stats["method"] == "pcb"
    # Positions 0-2 are unanimous -> the merged delta keeps the shared direction.
    assert merged["w"][0].item() > 0.5
    # Position 3 is 2-vs-1 -> the merged delta follows the majority.
    assert merged["w"][3].item() < 0.0


def test_pcb_merge_drops_all_but_the_kept_ratio():
    base = {"w": torch.zeros(1000)}
    g = torch.Generator().manual_seed(3)
    deltas = [{"w": torch.randn(1000, generator=g)} for _ in range(3)]
    merged, stats = pcb_merge(base, deltas, [1.0, 1.0, 1.0], ratio=0.1)
    # ~10% of the 3×1000 (task, parameter) scores survive the drop.
    assert 0.08 < stats["kept_fraction"] < 0.12
    # Positions where every task was dropped receive no update at all.
    assert (merged["w"] == 0).sum().item() > 0


def test_pcb_lambda_scales_the_merged_task_vector():
    base = {"w": torch.zeros(64)}
    g = torch.Generator().manual_seed(11)
    deltas = [{"w": torch.randn(64, generator=g)} for _ in range(2)]
    a, _ = pcb_merge(base, deltas, [1.0, 1.0], ratio=0.5, lam=1.0)
    b, _ = pcb_merge(base, deltas, [1.0, 1.0], ratio=0.5, lam=2.0)
    assert torch.allclose(b["w"], 2.0 * a["w"], atol=1e-6)


def test_pcb_merge_is_deterministic():
    base = {"w": torch.zeros(128)}
    g = torch.Generator().manual_seed(5)
    deltas = [{"w": torch.randn(128, generator=g)} for _ in range(3)]
    a, _ = pcb_merge(base, deltas, [1.0, 1.0, 1.0], ratio=0.2)
    b, _ = pcb_merge(base, deltas, [1.0, 1.0, 1.0], ratio=0.2)
    assert torch.equal(a["w"], b["w"])


def test_pcb_scope_tensor_applies_ratio_per_tensor():
    """Global ranking lets one tensor dominate the budget; per-tensor doesn't."""
    base = {"big": torch.zeros(2000), "small": torch.zeros(50)}
    g = torch.Generator().manual_seed(9)
    deltas = [
        {"big": torch.randn(2000, generator=g) * 10.0,
         "small": torch.randn(50, generator=g) * 0.001}
        for _ in range(2)
    ]
    per_tensor, st = pcb_merge(base, deltas, [1.0, 1.0], ratio=0.1, scope="tensor")
    # Per-tensor scoping guarantees the small tensor keeps ~10% of its own
    # entries rather than being crowded out by the large one.
    assert (per_tensor["small"] != 0).sum().item() > 0
    assert st["pcb_scope"] == "tensor"


def test_pcb_rejects_invalid_ratio():
    base = {"w": torch.zeros(4)}
    deltas = [{"w": torch.ones(4)}, {"w": torch.ones(4)}]
    with pytest.raises(Exception):
        pcb_merge(base, deltas, [1.0, 1.0], ratio=0.0)


def test_cosine_and_sign_agreement_bounds():
    a = torch.randn(1024)
    b = torch.randn(1024)
    cos = _cosine_similarity(a, b)
    sa = _sign_agreement(a, b)
    assert -1.0 <= cos <= 1.0
    assert 0.0 <= sa <= 1.0
    # Cosine of vector with itself = 1
    assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6
