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


def test_cosine_and_sign_agreement_bounds():
    a = torch.randn(1024)
    b = torch.randn(1024)
    cos = _cosine_similarity(a, b)
    sa = _sign_agreement(a, b)
    assert -1.0 <= cos <= 1.0
    assert 0.0 <= sa <= 1.0
    # Cosine of vector with itself = 1
    assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6
