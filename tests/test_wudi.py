"""Unit tests for WUDI-Merging (Cheng et al., ICML 2025).

These exercise the WUDI kernels on synthetic state dicts, without needing any
HuggingFace checkpoints.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mergese import (  # noqa: E402
    _is_linear_weight,
    _wudi_optimize_layer,
    wudi_merge,
)


def _wudi_objective(merged, taus, weights):
    """The interference objective WUDI minimises, for verification in tests."""
    total = 0.0
    for w, tau in zip(weights, taus):
        residual = merged - tau
        projection = residual @ tau.transpose(-1, -2)
        norm_sq = tau.pow(2).sum().clamp(min=1e-12)
        total += w * float(projection.pow(2).sum() / norm_sq)
    return total


# ---------------------------------------------------------------------------
# linear-layer detection
# ---------------------------------------------------------------------------

def test_is_linear_weight_matches_2d_weights():
    assert _is_linear_weight("encoder.layer.0.attention.self.query.weight",
                             torch.zeros(8, 8))
    assert _is_linear_weight("classifier.dense.weight", torch.zeros(4, 8))


def test_is_linear_weight_excludes_non_linear():
    # 1-D tensors (biases) are not linear weight matrices
    assert not _is_linear_weight("classifier.bias", torch.zeros(8))
    # embeddings and layer norms are excluded by name even when 2-D
    assert not _is_linear_weight("embeddings.word_embeddings.weight",
                                 torch.zeros(100, 8))
    assert not _is_linear_weight("encoder.layer.0.output.LayerNorm.weight",
                                 torch.zeros(8))


# ---------------------------------------------------------------------------
# per-layer optimisation
# ---------------------------------------------------------------------------

def test_wudi_layer_does_not_increase_objective():
    """Optimisation must not raise the interference objective above its value
    at the weighted-sum initialisation."""
    g = torch.Generator().manual_seed(0)
    taus = [torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g)]
    weights = [0.5, 0.5]

    init = torch.zeros_like(taus[0])
    for w, tau in zip(weights, taus):
        init = init + w * tau

    merged = _wudi_optimize_layer(taus, weights, num_steps=200, lr=1e-2,
                                  device=torch.device("cpu"))
    assert _wudi_objective(merged, taus, weights) <= _wudi_objective(init, taus, weights) + 1e-6


def test_wudi_layer_recovers_single_task():
    """With one task, the merged delta should stay at that task vector."""
    tau = torch.randn(6, 6, generator=torch.Generator().manual_seed(1))
    merged = _wudi_optimize_layer([tau], [1.0], num_steps=100, lr=1e-3,
                                  device=torch.device("cpu"))
    assert torch.allclose(merged, tau, atol=1e-4)


def test_wudi_layer_is_deterministic():
    g = torch.Generator().manual_seed(2)
    taus = [torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g)]
    a = _wudi_optimize_layer(taus, [1.0, 1.0], num_steps=50, lr=1e-3,
                             device=torch.device("cpu"))
    b = _wudi_optimize_layer(taus, [1.0, 1.0], num_steps=50, lr=1e-3,
                             device=torch.device("cpu"))
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# full merge
# ---------------------------------------------------------------------------

def test_wudi_merge_routes_linear_and_non_linear():
    base = {
        "encoder.weight": torch.zeros(8, 8),
        "encoder.bias":   torch.zeros(8),
    }
    d1 = {
        "encoder.weight": torch.randn(8, 8, generator=torch.Generator().manual_seed(3)),
        "encoder.bias":   torch.ones(8),
    }
    d2 = {
        "encoder.weight": torch.randn(8, 8, generator=torch.Generator().manual_seed(4)),
        "encoder.bias":   torch.ones(8) * 3.0,
    }
    merged, stats = wudi_merge(base, [d1, d2], [1.0, 1.0], num_steps=25, lr=1e-3)

    assert stats["method"] == "wudi"
    assert stats["wudi_linear_layers"] == 1
    assert stats["averaged_layers"] == 1
    # Bias is the weighted average of the two deltas: (1 + 3) / 2 = 2
    assert torch.allclose(merged["encoder.bias"], torch.ones(8) * 2.0)
    # Encoder weight moved away from the zero base
    assert not torch.allclose(merged["encoder.weight"], base["encoder.weight"])


def test_wudi_merge_preserves_base_dtype():
    base = {"encoder.weight": torch.zeros(4, 4, dtype=torch.float32)}
    d1 = {"encoder.weight": torch.randn(4, 4)}
    d2 = {"encoder.weight": torch.randn(4, 4)}
    merged, _ = wudi_merge(base, [d1, d2], [1.0, 1.0], num_steps=10, lr=1e-3)
    assert merged["encoder.weight"].dtype == torch.float32


def test_wudi_merge_normalises_weights():
    """Non-normalised weights are rescaled to sum to 1 (matches average_merge)."""
    base = {"encoder.bias": torch.zeros(4)}
    d1 = {"encoder.bias": torch.ones(4) * 2.0}
    d2 = {"encoder.bias": torch.ones(4) * 4.0}
    merged, stats = wudi_merge(base, [d1, d2], [2.0, 2.0], num_steps=1, lr=1e-3)
    # Equal (rescaled) weights -> mean of 2 and 4 = 3
    assert torch.allclose(merged["encoder.bias"], torch.ones(4) * 3.0)
    assert pytest.approx(sum(stats["weights"]), rel=1e-6) == 1.0
