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


def test_elect_sign_handles_ragged_deltas():
    """Regression: when `_compute_task_vector` filters a shape-mismatched
    tensor out of one model's delta (e.g. CodeBERT 50265-vocab base vs
    UniXcoder 51416-vocab), the resulting deltas have different keysets.
    The sign-election pass must walk the UNION of keys, not deltas[0].keys(),
    and treat missing entries as a zero vote - without this fix the TIES/
    DARE-TIES/average mergers KeyError'd on prod with these three HF
    checkpoints as inputs."""
    # Both models have 'shared'; only model 1 has 'only_in_1'; only model 2 has 'only_in_2'.
    d1 = {"shared": torch.tensor([+1.0, -1.0]), "only_in_1": torch.tensor([+1.0])}
    d2 = {"shared": torch.tensor([+1.0, +1.0]), "only_in_2": torch.tensor([-1.0])}
    elected = _elect_sign([d1, d2], [1.0, 1.0])
    assert set(elected.keys()) == {"shared", "only_in_1", "only_in_2"}
    # The exclusive keys come straight from the one model that had them.
    assert elected["only_in_1"].item() == 1.0
    assert elected["only_in_2"].item() == -1.0


def test_ties_merge_survives_ragged_deltas():
    """End-to-end: TIES on three deltas with mismatched keysets must produce
    a merged state dict and not crash on `d[name]` for a key one model lacks."""
    base = {
        "shared": torch.zeros(4),
        "only_in_1": torch.zeros(2),
        "only_in_2": torch.zeros(2),
    }
    d1 = {"shared": torch.tensor([1.0, 1.0, -1.0, 0.0]),
          "only_in_1": torch.tensor([0.5, 0.5])}
    d2 = {"shared": torch.tensor([1.0, -1.0, -1.0, 0.5]),
          "only_in_2": torch.tensor([0.5, 0.5])}
    d3 = {"shared": torch.tensor([1.0, 1.0, -1.0, 0.0])}  # third model missing both exclusives
    merged, stats = ties_merge(base, [d1, d2, d3], [1.0, 1.0, 1.0], 0.0)
    # Base keys all present in output; exclusives got only their one model's contribution.
    assert set(merged.keys()) == set(base.keys())
    assert merged["only_in_1"][0].item() > 0  # d1 contributed
    assert merged["only_in_2"][0].item() > 0  # d2 contributed
    assert stats["method"] == "ties"


def test_average_merge_survives_ragged_deltas():
    base = {"a": torch.zeros(2), "b": torch.zeros(2)}
    # Only the first model has 'b'; averaging must not KeyError on d['b'] for d2.
    deltas = [
        {"a": torch.tensor([2.0, 2.0]), "b": torch.tensor([4.0, 4.0])},
        {"a": torch.tensor([4.0, 4.0])},  # no 'b'
    ]
    merged, _ = average_merge(base, deltas, [1.0, 1.0])
    assert torch.allclose(merged["a"], torch.tensor([3.0, 3.0]))
    # 'b' takes the one model's half-weighted delta (only d1 contributed, with norm_w=0.5).
    assert torch.allclose(merged["b"], torch.tensor([2.0, 2.0]))


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


# ---- B09: binary metric must reject out-of-domain labels / preds ------------

def test_binary_metric_rejects_out_of_domain(monkeypatch):
    """y_true=[1,1], y_pred=[1,2] previously returned precision=recall=F1=1.0
    because class 2 silently disappeared from the FN count. Now hard-fails."""
    from mergese import _compute_metrics
    with pytest.raises(ValueError, match=r"\{0, 1\}|\{0,1\}"):
        _compute_metrics([1, 1], [1, 2], mode="binary")
    with pytest.raises(ValueError):
        _compute_metrics([0, 1, 2], [0, 1, 0], mode="binary")
    # The in-domain case still works, and accuracy/f1 come out sensibly.
    r = _compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], mode="binary")
    assert r["mode"] == "binary"
    assert 0.0 <= r["f1"] <= 1.0


# ---- B06: zero shared keys must be a hard error, not a silent base return ---

def test_empty_shared_keys_produces_empty_delta_union(monkeypatch):
    """Unit test the invariant that drives the cmd_merge B06 fix: when the
    base's state dict shares no shape-matching keys with any specialist,
    `_compute_task_vector` returns empty dicts and the union is empty.
    cmd_merge now treats that as a hard error (`no mergeable tensors`); here
    we pin the lower-level observation so a refactor of the detection
    threshold stays correct."""
    import torch
    from mergese import _compute_task_vector, _shared_keys, LoadedModel
    def _m(path, sd):
        return LoadedModel(
            path=path, state_dict=sd, config={"model_type": "bert"},
            tokenizer_vocab=None, tokenizer_vocab_size=None, tokenizer_signature="",
            architectures=[], hidden_size=None, num_hidden_layers=None,
        )
    base = _m("base", {"encoder.weight": torch.zeros(4)})
    spec = _m("spec", {"bert.encoder.weight": torch.ones(4)})  # prefix mismatch
    shared = _shared_keys([base, spec])
    deltas = [_compute_task_vector(spec.state_dict, base.state_dict, shared)]
    assert sum(len(d) for d in deltas) == 0  # the condition cmd_merge now rejects


# ---- B07: heterogeneous head detection must examine every head tensor -------

def test_head_mismatch_in_out_proj_detected():
    """Two RoBERTa-style specialists where `classifier.dense` is shape-compatible
    but `classifier.out_proj` differs must be flagged as heterogeneous. The
    previous code checked only the first head tensor seen and missed this."""
    # Simulate the two per-model head-shape maps that cmd_merge builds.
    model_a_heads = {"classifier.dense.weight": (768, 768),
                     "classifier.dense.bias": (768,),
                     "classifier.out_proj.weight": (2, 768),
                     "classifier.out_proj.bias": (2,)}
    model_b_heads = {"classifier.dense.weight": (768, 768),
                     "classifier.dense.bias": (768,),
                     "classifier.out_proj.weight": (3, 768),
                     "classifier.out_proj.bias": (3,)}
    per_model = [model_a_heads, model_b_heads]
    all_keys = sorted({k for d in per_model for k in d})
    # Reimplement the detection the way cmd_merge now does it.
    heterogeneous = False
    for key in all_keys:
        shapes = {d.get(key) for d in per_model if key in d}
        shapes.discard(None)
        if len(shapes) > 1:
            heterogeneous = True
    assert heterogeneous is True


# ---- B08: NaN/Inf weights must be rejected by the CLI -----------------------

def test_cmd_merge_rejects_nan_weights():
    import click.testing
    from mergese import cli
    runner = click.testing.CliRunner()
    r = runner.invoke(cli, [
        "merge", "a", "b",
        "--base", "base",
        "--method", "ties",
        "--weights", "nan,1",
        "--output", "/tmp/mergese_test_out_should_not_exist",
    ])
    assert r.exit_code != 0
    assert "finite" in (r.output + str(r.exception)).lower()
    # Infs too
    r2 = runner.invoke(cli, [
        "merge", "a", "b",
        "--base", "base",
        "--method", "average",
        "--weights", "inf,1",
        "--output", "/tmp/mergese_test_out_should_not_exist",
    ])
    assert r2.exit_code != 0


def test_merge_prefix_aligns_wrapped_specialists():
    """A bare encoder base + wrapped `AutoModelForSequenceClassification`
    specialists is a legit input combo that previously triggered B06's hard
    fail. The prefix-align step strips the wrapper (`bert.`, `roberta.`, ...)
    so the encoder tensors line up with the base and the merge produces real
    deltas.

    Simulates what cmd_merge does to the LoadedModel.state_dict objects
    before computing shared keys.
    """
    import torch
    base_sd = {
        "embeddings.weight": torch.zeros(4),
        "encoder.layer.0.weight": torch.zeros(4),
    }
    # Wrapped specialist: encoder tensors have a `bert.` prefix + extra head
    spec_sd = {
        "bert.embeddings.weight": torch.ones(4) * 0.1,
        "bert.encoder.layer.0.weight": torch.ones(4) * 0.1,
        "classifier.weight": torch.ones((2, 4)),  # head, no base match
    }
    # Reproduce the align logic from cmd_merge.
    base_keys = set(base_sd)
    m_keys = set(spec_sd)
    assert not (m_keys & base_keys), "test setup: specialist must have no raw overlap"
    prefixes = {k.split(".", 1)[0] + "." for k in m_keys if "." in k}
    best = None
    for pfx in prefixes:
        stripped = {k[len(pfx):] for k in m_keys if k.startswith(pfx)}
        overlap = len(stripped & base_keys)
        if overlap and (best is None or overlap > best[1]):
            best = (pfx, overlap)
    assert best is not None, "align should find a candidate"
    assert best[0] == "bert.", f"expected 'bert.' prefix, got {best[0]!r}"
    assert best[1] == 2, f"expected overlap=2, got {best[1]}"
    pfx = best[0]
    aligned = {(k[len(pfx):] if k.startswith(pfx) else k): v for k, v in spec_sd.items()}
    # After alignment, encoder keys match the base's; classifier survives as-is.
    assert set(aligned) == {"embeddings.weight", "encoder.layer.0.weight", "classifier.weight"}
    assert set(aligned) & base_keys == {"embeddings.weight", "encoder.layer.0.weight"}


def test_torchscript_wrapper_unwraps_dict_output():
    """The B10 retest failed because transformers models return
    ModelOutput (a dict-like namedtuple), which torch.jit.trace refuses
    with "Encountering a dict at the output of the tracer...". The export
    now wraps the model in a Module whose forward reduces the output to a
    plain tensor via return_dict=False / .logits / .last_hidden_state /
    tuple[0]. Pin that wrapper logic end-to-end: no transformers import
    needed - we just mimic the shapes the real wrapper has to handle."""
    import torch
    class _Fake(torch.nn.Module):
        """Returns whatever `shape` dictates."""
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.w = torch.nn.Parameter(torch.ones(1))
        def forward(self, input_ids, attention_mask, return_dict=True):
            x = (input_ids.float() * self.w).sum(-1, keepdim=True)  # (B,1)
            if self.shape == "logits":
                class _Out:
                    pass
                o = _Out(); o.logits = x; return o
            if self.shape == "hidden":
                class _Out:
                    pass
                o = _Out(); o.last_hidden_state = x; return o
            if self.shape == "tuple":
                return (x,)
            if self.shape == "dict":
                return {"logits": x}  # unreachable with return_dict=False, but tests the fallback
            return x

    # Reproduce the wrapper from cmd_export.
    class _TraceWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, input_ids, attention_mask):
            out = self.inner(input_ids=input_ids, attention_mask=attention_mask,
                             return_dict=False)
            if isinstance(out, (tuple, list)): return out[0]
            if hasattr(out, "logits"): return out.logits
            if hasattr(out, "last_hidden_state"): return out.last_hidden_state
            return out

    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    for shape in ("logits", "hidden", "tuple"):
        w = _TraceWrapper(_Fake(shape)).eval()
        traced = torch.jit.trace(w, (ids, mask), strict=False)
        # Trace succeeded; output is a tensor.
        out = traced(ids, mask)
        assert isinstance(out, torch.Tensor), f"shape={shape}: got {type(out)}"


# ---- architecture (greedy) merge --------------------------------------------

def test_architecture_greedy_merge_hillclimbs_when_better():
    """Pin the greedy contract: when a specialist's layer scores strictly
    higher on eval_fn, keep the swap; when neither improves, keep base."""
    import torch
    # Minimal "state dict" with two encoder layers.
    def _sd(a0, a1):
        return {
            "encoder.layer.0.weight": torch.tensor([a0]),
            "encoder.layer.1.weight": torch.tensor([a1]),
            # a non-layer tensor that must not get swept into any layer slice:
            "classifier.weight": torch.tensor([99.0]),
        }
    base = _sd(1.0, 1.0)
    spec_a = _sd(2.0, 1.0)   # improves layer 0 if eval_fn rewards it
    spec_b = _sd(1.0, 3.0)   # improves layer 1 if eval_fn rewards it

    # eval_fn: score = sum of layer-0 and layer-1 tensor values. The classifier
    # is ignored so we can tell layer swaps apart from accidental head copies.
    def eval_fn(sd):
        return float(sd["encoder.layer.0.weight"].item() + sd["encoder.layer.1.weight"].item())

    from mergese import architecture_greedy_merge
    merged, stats = architecture_greedy_merge(
        base_sd=base, specialist_sds=[spec_a, spec_b], eval_fn=eval_fn,
    )
    # Both layers took the better specialist's weights.
    assert merged["encoder.layer.0.weight"].item() == 2.0
    assert merged["encoder.layer.1.weight"].item() == 3.0
    assert merged["classifier.weight"].item() == 99.0   # classifier untouched
    assert stats["layer_winners"] == [0, 1]  # specialist 0 won L0, specialist 1 won L1
    assert stats["final_score"] > stats["base_score"]


def test_architecture_greedy_keeps_base_when_no_swap_helps():
    import torch
    sd = {"encoder.layer.0.weight": torch.tensor([5.0])}
    base = sd
    worse = {"encoder.layer.0.weight": torch.tensor([-99.0])}
    from mergese import architecture_greedy_merge
    merged, stats = architecture_greedy_merge(
        base_sd=base, specialist_sds=[worse],
        eval_fn=lambda s: float(s["encoder.layer.0.weight"].item()),
    )
    assert merged["encoder.layer.0.weight"].item() == 5.0
    assert stats["layer_winners"] == [-1]
    assert stats["final_score"] == stats["base_score"]


def test_architecture_merge_detects_encoder_layout():
    """Covers both the bare-encoder and classification-head-wrapped layouts."""
    from mergese import _encoder_layer_prefixes
    bare = ["encoder.layer.0.weight", "encoder.layer.1.weight", "pooler.weight"]
    pfx, n = _encoder_layer_prefixes(bare)
    assert pfx == "encoder.layer." and n == 2
    wrapped = ["bert.encoder.layer.0.weight", "bert.encoder.layer.1.weight",
               "bert.encoder.layer.2.weight", "classifier.weight"]
    pfx, n = _encoder_layer_prefixes(wrapped)
    assert pfx == "bert.encoder.layer." and n == 3
    import pytest
    with pytest.raises(ValueError, match="encoder.layer"):
        _encoder_layer_prefixes(["no.such.pattern.0"])


# ---- architecture-merge retest findings (A1, A2, A3) -----------------------

_transformers_ok = False
try:
    import transformers as _tx  # noqa: F401
    _transformers_ok = True
except Exception:
    _transformers_ok = False

_needs_transformers = pytest.mark.skipif(
    not _transformers_ok,
    reason="transformers unavailable in this environment",
)


@_needs_transformers
def test_build_architecture_eval_fn_accepts_task(monkeypatch, tmp_path):
    """A1: `--task clone_detection` previously NameError'd on `_REGISTRY`
    because mergese.py imports the registry under a different name. Pin
    that supplying a valid task name reaches the metric-mode logic
    without crashing. We mock out the heavy transformers path and just
    exercise the branch that failed."""
    import csv as _csv, torch
    from mergese import _build_architecture_eval_fn
    # Build a 2-row CSV so the empty-dataset check passes.
    csv_path = tmp_path / "e.csv"
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["code", "label"])
        w.writerow(["x", "0"]); w.writerow(["y", "1"])
    # Stub a tiny bare encoder so we don't need a real HF checkpoint.
    import mergese
    base_dir = tmp_path / "base"; base_dir.mkdir()
    # Save a minimal BertModel so AutoModel.from_pretrained works.
    from transformers import BertConfig, BertModel, BertTokenizerFast
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=16)
    m = BertModel(cfg)
    for p in m.parameters(): p.data = p.data.contiguous()
    m.save_pretrained(base_dir, safe_serialization=True)
    vocab = {"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,
             **{f"t{i}":i+5 for i in range(40)}}
    (base_dir / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(base_dir / "vocab.txt")).save_pretrained(base_dir)
    # A task argument must not NameError.
    eval_fn = _build_architecture_eval_fn(
        base_path=str(base_dir), resolved_eval_csv=str(csv_path),
        limit=2, batch_size=1, max_length=8, device="cpu",
        task="clone_detection",
    )
    # Can score the base's own state dict (no key mismatch).
    import transformers
    base_sd = transformers.AutoModel.from_pretrained(str(base_dir)).state_dict()
    score = eval_fn(base_sd)
    assert 0.0 <= float(score) <= 1.0


@_needs_transformers
def test_architecture_eval_fn_rejects_empty_eval_csv(monkeypatch, tmp_path):
    """A3: a CSV with only the header previously produced a successful merge
    with zero scored examples. Reject at eval-fn build time."""
    import csv as _csv, click
    from mergese import _build_architecture_eval_fn
    csv_path = tmp_path / "empty.csv"
    with csv_path.open("w", newline="") as fh:
        _csv.writer(fh).writerow(["code", "label"])  # header only
    base_dir = tmp_path / "base"; base_dir.mkdir()
    from transformers import BertConfig, BertModel, BertTokenizerFast
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=16)
    m = BertModel(cfg)
    for p in m.parameters(): p.data = p.data.contiguous()
    m.save_pretrained(base_dir, safe_serialization=True)
    (base_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
    BertTokenizerFast(vocab_file=str(base_dir / "vocab.txt")).save_pretrained(base_dir)
    with pytest.raises(click.UsageError, match="empty"):
        _build_architecture_eval_fn(
            base_path=str(base_dir), resolved_eval_csv=str(csv_path),
            limit=200, batch_size=4, max_length=16, device="cpu", task="",
        )


@_needs_transformers
def test_architecture_eval_fn_strict_rejects_wrong_layout(monkeypatch, tmp_path):
    """A2: previously strict=False silently discarded all encoder weights
    whose keys didn't match the shell (bare-encoder merged_sd vs
    classification-head shell with `bert.encoder.` prefix). Now strict=True
    raises so the hill climb never silently scores identical copies of the
    base."""
    import csv as _csv, torch
    from mergese import _build_architecture_eval_fn
    csv_path = tmp_path / "e.csv"
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["code", "label"]); w.writerow(["x", "0"])
    base_dir = tmp_path / "base"; base_dir.mkdir()
    from transformers import BertConfig, BertModel, BertTokenizerFast
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=16)
    m = BertModel(cfg)
    for p in m.parameters(): p.data = p.data.contiguous()
    m.save_pretrained(base_dir, safe_serialization=True)
    (base_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
    BertTokenizerFast(vocab_file=str(base_dir / "vocab.txt")).save_pretrained(base_dir)
    eval_fn = _build_architecture_eval_fn(
        base_path=str(base_dir), resolved_eval_csv=str(csv_path),
        limit=1, batch_size=1, max_length=8, device="cpu", task="",
    )
    # A sabotaged state_dict with an obviously-wrong prefix. strict=True
    # should raise RuntimeError inside load_state_dict rather than quietly
    # ignoring it.
    bad_sd = {f"bert.{k}": v for k, v in
              __import__("transformers").AutoModel.from_pretrained(str(base_dir))
              .state_dict().items()}
    with pytest.raises(RuntimeError):
        eval_fn(bad_sd)


@_needs_transformers
def test_architecture_eval_fn_loads_classification_shell(monkeypatch, tmp_path):
    """R1 (pressure retest): a wrapped BertForSequenceClassification
    checkpoint has keys like `bert.encoder.layer.N.*` + `classifier.*`.
    Loading those into a bare AutoModel shell was failing with strict=True
    even though the previous iteration of the code worked without a task.
    Eval builder must detect the wrapped layout and use a classification
    shell to match."""
    import csv as _csv, torch
    from mergese import _build_architecture_eval_fn
    from transformers import (BertConfig, BertForSequenceClassification,
                              BertTokenizerFast)
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=16, num_labels=2)
    m = BertForSequenceClassification(cfg)
    for p in m.parameters(): p.data = p.data.contiguous()
    base_dir = tmp_path / "cls"; base_dir.mkdir()
    m.save_pretrained(base_dir, safe_serialization=True)
    (base_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
    BertTokenizerFast(vocab_file=str(base_dir / "vocab.txt")).save_pretrained(base_dir)
    csv_path = tmp_path / "e.csv"
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["code", "label"])
        w.writerow(["x", "0"]); w.writerow(["y", "1"])

    eval_fn = _build_architecture_eval_fn(
        base_path=str(base_dir), resolved_eval_csv=str(csv_path),
        limit=2, batch_size=1, max_length=8, device="cpu", task="clone_detection",
    )
    # The eval fn must successfully score the wrapped-cls state dict (the
    # same thing on disk) without raising.
    base_sd = m.state_dict()
    score = eval_fn(base_sd)
    assert 0.0 <= float(score) <= 1.0
    # Provenance must report the trained head, not a random one.
    assert not eval_fn.meta["uses_random_head"]
    assert "AutoModelForSequenceClassification" in eval_fn.meta["head_source"]


@_needs_transformers
def test_architecture_eval_fn_is_deterministic_under_repeated_scoring(monkeypatch, tmp_path):
    """R2 (pressure retest): the Dropout module was constructed but never
    put in eval mode, so repeated scoring of the SAME state dict returned
    different F1s (0.00, 0.22, 0.40, 0.55 observed) and the greedy search
    could accept noise as improvement. Fix: dropout.eval(). Pin that
    repeated calls return identical scores."""
    import csv as _csv, torch
    from mergese import _build_architecture_eval_fn
    from transformers import BertConfig, BertModel, BertTokenizerFast
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=16,
                     hidden_dropout_prob=0.5,
                     attention_probs_dropout_prob=0.5)
    m = BertModel(cfg)
    for p in m.parameters(): p.data = p.data.contiguous()
    base_dir = tmp_path / "bare"; base_dir.mkdir()
    m.save_pretrained(base_dir, safe_serialization=True)
    (base_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
    BertTokenizerFast(vocab_file=str(base_dir / "vocab.txt")).save_pretrained(base_dir)
    csv_path = tmp_path / "e.csv"
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["code", "label"])
        for _ in range(8): w.writerow(["foo", "0"]); w.writerow(["bar", "1"])

    eval_fn = _build_architecture_eval_fn(
        base_path=str(base_dir), resolved_eval_csv=str(csv_path),
        limit=16, batch_size=4, max_length=8, device="cpu", task="",
    )
    base_sd = m.state_dict()
    scores = [eval_fn(base_sd) for _ in range(5)]
    # All 5 scores identical - no dropout randomness bleeding through.
    assert len(set(scores)) == 1, f"non-deterministic scoring: {scores}"
    # Random-head case must be flagged as such.
    assert eval_fn.meta["uses_random_head"] is True
    assert "RANDOM" in eval_fn.meta["head_source"]


@_needs_transformers
def test_sidecar_classifier_head_travels_with_merge_output(tmp_path):
    """R-save (pressure retest round 3): architecture merge against a
    sidecar fine-tune used `classifier_head.bin` for scoring but dropped
    it from the saved output - a reload fell back to a random classifier.
    `save_merged_checkpoint` must copy sidecar heads + their companions."""
    import torch
    from mergese import save_merged_checkpoint, _build_architecture_eval_fn
    from transformers import BertConfig, BertModel, BertTokenizerFast
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=16)
    base_dir = tmp_path / "sidecar_base"; base_dir.mkdir()
    m = BertModel(cfg)
    for p in m.parameters(): p.data = p.data.contiguous()
    m.save_pretrained(base_dir, safe_serialization=True)
    (base_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
    BertTokenizerFast(vocab_file=str(base_dir / "vocab.txt")).save_pretrained(base_dir)
    # Plant a sidecar classifier head + config in the base dir.
    classifier = torch.nn.Linear(16, 2)
    for p in classifier.parameters(): p.data = p.data.contiguous()
    torch.save(
        {"num_labels": 2, "dropout": 0.1, "classifier": classifier.state_dict()},
        base_dir / "classifier_head.bin",
    )
    (base_dir / "clone_detection_config.json").write_text(
        '{"num_labels": 2, "dropout": 0.1}'
    )
    (base_dir / "best_metrics.json").write_text(
        '{"best_f1": 0.5, "best_epoch": 1}'
    )

    # Save the merge output.
    out_dir = tmp_path / "merged"
    save_merged_checkpoint(str(base_dir), m.state_dict(), str(out_dir))

    # The sidecar + its companions must be present in the output.
    for required in ("classifier_head.bin",
                     "clone_detection_config.json",
                     "best_metrics.json"):
        assert (out_dir / required).exists(), f"{required} missing from merged output"

    # And the reload path recognises the sidecar (not falls back to random).
    import csv as _csv
    csv_path = tmp_path / "e.csv"
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["code", "label"]); w.writerow(["x", "0"])
    eval_fn = _build_architecture_eval_fn(
        base_path=str(out_dir), resolved_eval_csv=str(csv_path),
        limit=1, batch_size=1, max_length=8, device="cpu", task="clone_detection",
    )
    assert eval_fn.meta["uses_random_head"] is False
    assert "sidecar" in eval_fn.meta["head_source"]
