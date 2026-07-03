"""
MergeSE - Model Merging for Software Engineering
=================================================

A general-purpose CLI for merging fine-tuned HuggingFace encoder
checkpoints across the full spectrum of SE classification tasks:

    * Code clone detection             (BigCloneBench, CLCDSA, GPTCloneBench, POJ-104)
    * Vulnerability detection          (Devign, ReVeal, Big-Vul, D2A, Draper)
    * Defect / bug prediction          (Defects4J, PROMISE, CodeXGLUE-Defect)
    * Code-smell detection             (MLCQ, Qualitas)
    * Commit classification            (CommitBench)
    * Code-review acceptability        (CodeReview)
    * Comment-code consistency
    * Exception-type prediction        (CodeXGLUE-Exception)
    * Type inference (closed-set)      (Typilus, Type4Py, ManyTypes4Py)
    * Any custom binary / multi-class CSV

The merging itself is fully task-agnostic - it operates on state-dict tensors,
so models trained on *different* SE tasks can be merged provided they share a
base encoder + tokenizer. When the classifier heads differ in shape, MergeSE
either skips them (`--encoder-only`, default) or warns and merges per-tensor.

Implements:
    * TIES-Merging (Yadav et al., NeurIPS 2023)
    * DARE-TIES   (Yu et al., ICML 2024 + TIES)
    * WUDI-Merging (Cheng et al., ICML 2025)
    * Simple task-vector averaging

Commands
--------
    mergese tasks     List supported SE tasks and benchmarks
    mergese inspect   Analyse checkpoint compatibility
    mergese merge     Merge checkpoints (TIES / DARE-TIES / WUDI / average)
    mergese evaluate  Evaluate a model on any registered SE benchmark
    mergese export    Export a merged model (HF / ONNX / TorchScript)

Tool artifact for the ASE 2026 Tool-Track submission:
    "MergeSE: Post-hoc Model Merging for Software Engineering Tasks
     Without Retraining"

Built on the model-merging methodology of our research-track submission:
    "A Unified Model for Cross-Domain Clone Detection via Model Merging"

Palash R. Roy, Banani Roy, Chanchal K. Roy, Kevin A. Schneider
University of Saskatchewan
"""

# pip install click torch transformers rich safetensors tqdm

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import click

from mergese_tasks import (
    TaskSpec,
    all_tasks,
    detect_input_kind,
    get as get_task,
    names as task_names,
    pick_metric,
)

__version__ = "0.2.0"

# Heuristics for "classifier head" tensors that are tied to a specific task
# (number of labels) and therefore not safely mergeable across tasks.
_CLASSIFIER_HEAD_PATTERNS = (
    "classifier.",          # BertForSequenceClassification, Roberta, ...
    "score.",               # LLaMA classifier
    ".classifier.",
    "qa_outputs.",          # SQuAD-style heads
    "logits_proj.",
)


def _is_classifier_head(name: str) -> bool:
    """Heuristic: is this tensor part of a downstream classifier head?"""
    return any(p in name for p in _CLASSIFIER_HEAD_PATTERNS)

# ---------------------------------------------------------------------------
# Lazy heavy imports (torch / transformers / rich) so that `--help` is fast
# ---------------------------------------------------------------------------

def _lazy_torch():
    import torch  # noqa: F401
    return torch


def _lazy_transformers():
    import transformers  # noqa: F401
    return transformers


def _lazy_rich():
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
    )
    return Console, Table, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("mergese")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)


def _configure_verbosity(verbose: bool) -> None:
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch = _lazy_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """A loaded checkpoint along with its identifying metadata."""
    path: str
    state_dict: Dict[str, "torch.Tensor"]  # type: ignore[name-defined]
    config: dict
    tokenizer_vocab: Optional[Dict[str, int]]
    tokenizer_vocab_size: Optional[int]
    tokenizer_signature: Optional[str]
    architectures: List[str]
    hidden_size: Optional[int]
    num_hidden_layers: Optional[int]


def _resolve_path(path: str) -> str:
    """Treat existing local dirs as local paths; otherwise pass through (HF Hub id)."""
    p = Path(path).expanduser()
    if p.exists():
        return str(p.resolve())
    return path


def _load_state_dict(path: str) -> Dict[str, "torch.Tensor"]:  # type: ignore[name-defined]
    """Load a state dict from a local dir or HF Hub id (prefers safetensors)."""
    torch = _lazy_torch()
    transformers = _lazy_transformers()
    AutoModel = transformers.AutoModel

    local = Path(path)
    if local.is_dir():
        # Prefer safetensors when present
        st_files = sorted(local.glob("*.safetensors"))
        if st_files:
            try:
                from safetensors.torch import load_file
                sd: Dict[str, "torch.Tensor"] = {}
                for f in st_files:
                    sd.update(load_file(str(f)))
                return sd
            except ImportError:
                logger.warning("safetensors not installed; falling back to torch.load")

        bin_files = sorted(local.glob("pytorch_model*.bin"))
        if bin_files:
            sd = {}
            for f in bin_files:
                sd.update(torch.load(str(f), map_location="cpu"))
            return sd

        # Fall through to AutoModel.from_pretrained
        model = AutoModel.from_pretrained(str(local))
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # HF Hub id
    model = AutoModel.from_pretrained(path)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _load_config(path: str) -> dict:
    transformers = _lazy_transformers()
    AutoConfig = transformers.AutoConfig
    cfg = AutoConfig.from_pretrained(path)
    return cfg.to_dict()


def _load_tokenizer_info(path: str) -> Tuple[Optional[Dict[str, int]], Optional[int], Optional[str]]:
    """Return (vocab, vocab_size, signature_hash) - gracefully tolerate missing tokenizer."""
    try:
        transformers = _lazy_transformers()
        tok = transformers.AutoTokenizer.from_pretrained(path)
        vocab = tok.get_vocab()
        size = len(vocab)
        # Stable signature: hash the (sorted) first 200 ids + size
        sig_input = f"{size}|{sorted(vocab.items(), key=lambda kv: kv[1])[:200]}"
        import hashlib
        sig = hashlib.md5(sig_input.encode("utf-8")).hexdigest()[:12]
        return vocab, size, sig
    except Exception as e:
        logger.debug(f"tokenizer load failed for {path}: {e}")
        return None, None, None


def load_model(path: str, with_tokenizer: bool = True) -> LoadedModel:
    """Load a checkpoint (state dict + config + tokenizer info)."""
    resolved = _resolve_path(path)
    logger.debug(f"Loading model from {resolved}")
    sd = _load_state_dict(resolved)
    cfg = _load_config(resolved)
    if with_tokenizer:
        vocab, vsize, sig = _load_tokenizer_info(resolved)
    else:
        vocab, vsize, sig = None, None, None
    return LoadedModel(
        path=path,
        state_dict=sd,
        config=cfg,
        tokenizer_vocab=vocab,
        tokenizer_vocab_size=vsize,
        tokenizer_signature=sig,
        architectures=cfg.get("architectures", []) or [],
        hidden_size=cfg.get("hidden_size"),
        num_hidden_layers=cfg.get("num_hidden_layers"),
    )


# ---------------------------------------------------------------------------
# Compatibility analysis
# ---------------------------------------------------------------------------

def _shared_keys(models: Sequence[LoadedModel]) -> List[str]:
    keys: Optional[set] = None
    for m in models:
        s = set(m.state_dict.keys())
        keys = s if keys is None else keys & s
    return sorted(keys or set())


def _flatten_delta(delta: Dict[str, "torch.Tensor"]) -> "torch.Tensor":  # type: ignore[name-defined]
    torch = _lazy_torch()
    return torch.cat([v.flatten().float() for v in delta.values()])


def _compute_task_vector(
    model_sd: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
    base_sd: Dict[str, "torch.Tensor"],   # type: ignore[name-defined]
    shared_keys: Iterable[str],
    encoder_only: bool = False,
) -> Dict[str, "torch.Tensor"]:           # type: ignore[name-defined]
    """Compute Δ = θ_model - θ_base over shared keys with matching shape.

    When `encoder_only=True`, classifier-head tensors are excluded from the
    delta so that the merged model retains the base's head untouched. This
    is the right default when merging models trained on *different* SE
    tasks (e.g. a clone detector + a vulnerability detector), whose heads
    have incompatible num_labels.
    """
    delta = {}
    for k in shared_keys:
        if encoder_only and _is_classifier_head(k):
            continue
        if model_sd[k].shape != base_sd[k].shape:
            continue
        delta[k] = (model_sd[k].float() - base_sd[k].float()).cpu()
    return delta


def _cosine_similarity(a: "torch.Tensor", b: "torch.Tensor") -> float:  # type: ignore[name-defined]
    torch = _lazy_torch()
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 or nb.item() == 0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def _sign_agreement(a: "torch.Tensor", b: "torch.Tensor") -> float:  # type: ignore[name-defined]
    sa = a.sign()
    sb = b.sign()
    nz = (sa != 0) & (sb != 0)
    if nz.sum().item() == 0:
        return 0.0
    return float((sa[nz] == sb[nz]).float().mean().item())


def _verdict(models: Sequence[LoadedModel]) -> str:
    """COMPATIBLE / RISKY / INCOMPATIBLE."""
    # Tokenizer signatures
    sigs = [m.tokenizer_signature for m in models if m.tokenizer_signature]
    sizes = [m.tokenizer_vocab_size for m in models if m.tokenizer_vocab_size]
    if sigs and len(set(sigs)) > 1:
        return "INCOMPATIBLE"
    if sizes and len(set(sizes)) > 1:
        return "INCOMPATIBLE"

    # Same base? Use model_type + hidden_size + num_layers + (optional _name_or_path)
    base_sigs = {
        (
            m.config.get("model_type"),
            m.config.get("hidden_size"),
            m.config.get("num_hidden_layers"),
            m.config.get("vocab_size"),
        )
        for m in models
    }
    if len(base_sigs) > 1:
        return "RISKY"
    return "COMPATIBLE"


# ---------------------------------------------------------------------------
# Merging core
# ---------------------------------------------------------------------------

def _trim_by_percentile(
    delta: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
    percentile: float,
) -> Tuple[Dict[str, "torch.Tensor"], float]:  # type: ignore[name-defined]
    """Zero-out values whose |x| is below the per-tensor p-th percentile.

    Uses ``torch.kthvalue`` rather than ``torch.quantile`` because the latter
    has a hard input-size limit (≈16M elements) that is regularly exceeded by
    transformer embedding tables (e.g. 50265 × 768 ≈ 38M for RoBERTa).
    """
    torch = _lazy_torch()
    trimmed: Dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
    total = 0
    zeroed = 0
    q = max(0.0, min(1.0, percentile / 100.0))
    for name, t in delta.items():
        flat = t.abs().flatten()
        n = flat.numel()
        if n == 0 or q == 0:
            trimmed[name] = t
            total += n
            continue
        k = max(1, int(round(q * n)))
        # kthvalue is the k-th smallest (1-indexed); zero everything <= it so
        # roughly q·n elements are removed.
        thr = torch.kthvalue(flat, k).values.item()
        mask = t.abs() > thr
        trimmed[name] = t * mask
        total += n
        zeroed += int((~mask).sum().item())
    frac = (zeroed / total) if total else 0.0
    return trimmed, frac


def _dare_drop(
    delta: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
    drop_rate: float,
    generator: "torch.Generator",       # type: ignore[name-defined]
) -> Dict[str, "torch.Tensor"]:        # type: ignore[name-defined]
    """Bernoulli(1-p) mask + rescale by 1/(1-p)."""
    torch = _lazy_torch()
    out: Dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
    keep = 1.0 - drop_rate
    if keep <= 0:
        raise click.UsageError("drop_rate must be in [0, 1).")
    for name, t in delta.items():
        mask = torch.bernoulli(
            torch.full_like(t, keep), generator=generator
        )
        out[name] = (t * mask) / keep
    return out


def _elect_sign(
    deltas: Sequence[Dict[str, "torch.Tensor"]],  # type: ignore[name-defined]
    weights: Sequence[float],
) -> Dict[str, "torch.Tensor"]:                  # type: ignore[name-defined]
    """Per-parameter majority sign, weighted by magnitude * lambda_k."""
    torch = _lazy_torch()
    signs: Dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
    keys = deltas[0].keys()
    for name in keys:
        score = torch.zeros_like(deltas[0][name])
        for k, d in enumerate(deltas):
            score = score + weights[k] * d[name]
        signs[name] = score.sign()
    return signs


def _merge_with_signs(
    deltas: Sequence[Dict[str, "torch.Tensor"]],  # type: ignore[name-defined]
    weights: Sequence[float],
    elected: Dict[str, "torch.Tensor"],          # type: ignore[name-defined]
) -> Tuple[Dict[str, "torch.Tensor"], float]:    # type: ignore[name-defined]
    """Keep only entries whose sign matches the elected sign; average the rest."""
    torch = _lazy_torch()
    merged: Dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
    total = 0
    conflicts = 0
    for name in deltas[0].keys():
        agg = torch.zeros_like(deltas[0][name])
        kept_count = torch.zeros_like(deltas[0][name])
        for k, d in enumerate(deltas):
            agree = (d[name].sign() == elected[name]) & (elected[name] != 0)
            agg = agg + weights[k] * d[name] * agree.float()
            kept_count = kept_count + agree.float()
            conflicts += int(((d[name].sign() != elected[name]) & (d[name] != 0) & (elected[name] != 0)).sum().item())
            total += int((d[name] != 0).sum().item())
        denom = kept_count.clamp(min=1.0)
        # Weighted average over agreeing models (normalise by sum of weights of agreeing models)
        merged[name] = agg / denom
    conflict_frac = (conflicts / total) if total else 0.0
    return merged, conflict_frac


def ties_merge(
    base_sd: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
    task_vectors: Sequence[Dict[str, "torch.Tensor"]],  # type: ignore[name-defined]
    weights: Sequence[float],
    trim_percentile: float,
) -> Tuple[Dict[str, "torch.Tensor"], dict]:  # type: ignore[name-defined]
    """TIES: trim -> elect sign -> sign-consistent merge -> add to base."""
    trimmed_list = []
    trimmed_fracs = []
    for tv in task_vectors:
        tr, frac = _trim_by_percentile(tv, trim_percentile)
        trimmed_list.append(tr)
        trimmed_fracs.append(frac)
    elected = _elect_sign(trimmed_list, weights)
    merged_delta, conflict_frac = _merge_with_signs(trimmed_list, weights, elected)
    merged_sd = {k: base_sd[k] + merged_delta.get(k, 0) for k in base_sd}
    # Coerce dtype back to base
    for k in merged_sd:
        merged_sd[k] = merged_sd[k].to(base_sd[k].dtype)
    stats = {
        "method": "ties",
        "trim_percentile": trim_percentile,
        "trimmed_fraction": float(sum(trimmed_fracs) / max(len(trimmed_fracs), 1)),
        "sign_conflict_fraction": float(conflict_frac),
    }
    return merged_sd, stats


def dare_ties_merge(
    base_sd: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
    task_vectors: Sequence[Dict[str, "torch.Tensor"]],  # type: ignore[name-defined]
    weights: Sequence[float],
    trim_percentile: float,
    drop_rate: float,
    seed: int,
) -> Tuple[Dict[str, "torch.Tensor"], dict]:  # type: ignore[name-defined]
    """DARE-TIES: random drop + rescale, then TIES."""
    torch = _lazy_torch()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    dared = [_dare_drop(tv, drop_rate, gen) for tv in task_vectors]
    merged_sd, stats = ties_merge(base_sd, dared, weights, trim_percentile)
    stats["method"] = "dare-ties"
    stats["drop_rate"] = drop_rate
    stats["seed"] = seed
    return merged_sd, stats


def average_merge(
    base_sd: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
    task_vectors: Sequence[Dict[str, "torch.Tensor"]],  # type: ignore[name-defined]
    weights: Sequence[float],
) -> Tuple[Dict[str, "torch.Tensor"], dict]:  # type: ignore[name-defined]
    """Plain task-vector averaging."""
    torch = _lazy_torch()
    keys = task_vectors[0].keys()
    w_sum = float(sum(weights)) or 1.0
    norm_w = [w / w_sum for w in weights]
    merged_delta: Dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
    for name in keys:
        agg = torch.zeros_like(task_vectors[0][name])
        for k, d in enumerate(task_vectors):
            agg = agg + norm_w[k] * d[name]
        merged_delta[name] = agg
    merged_sd = {k: base_sd[k] + merged_delta.get(k, 0) for k in base_sd}
    for k in merged_sd:
        merged_sd[k] = merged_sd[k].to(base_sd[k].dtype)
    return merged_sd, {"method": "average", "weights": list(norm_w)}


# ---------------------------------------------------------------------------
# WUDI merging (Cheng et al., ICML 2025)
# ---------------------------------------------------------------------------

WUDI_DEFAULT_STEPS = 300
WUDI_DEFAULT_LR = 1e-5

# Parameter-name hints for tensors that are *not* linear-layer weight matrices.
# WUDI's interference-minimisation objective is defined over linear layers, so
# these tensors (embeddings, LayerNorm, biases) fall back to weighted averaging.
_WUDI_NON_LINEAR_HINTS = (
    "embedding",
    "embeddings",
    "LayerNorm",
    "layer_norm",
    "ln_",
    "norm.",
    "position_ids",
)


def _is_linear_weight(name: str, tensor: "torch.Tensor") -> bool:  # type: ignore[name-defined]
    """True for 2-D linear-layer weight matrices - the tensors WUDI optimises."""
    if not name.endswith(".weight"):
        return False
    if tensor.ndim != 2:
        return False
    return not any(hint in name for hint in _WUDI_NON_LINEAR_HINTS)


def _wudi_optimize_layer(
    task_deltas: Sequence["torch.Tensor"],  # type: ignore[name-defined]
    weights: Sequence[float],
    num_steps: int,
    lr: float,
    device: "torch.device",                 # type: ignore[name-defined]
) -> "torch.Tensor":                        # type: ignore[name-defined]
    """Minimise cross-task interference for one linear layer.

    For task deltas τ₁...τ_K, the merged delta τ̂ is optimised so that the
    projection of each residual (τ̂ - τ_k) onto τ_k's row space is small,
    normalised by ‖τ_k‖²_F to balance tasks of different magnitudes. τ̂ is
    initialised at the weighted task-vector sum and refined with Adam. No task
    data is required.
    """
    torch = _lazy_torch()
    taus = [t.to(device=device, dtype=torch.float32) for t in task_deltas]

    merged = torch.zeros_like(taus[0])
    for w, tau in zip(weights, taus):
        merged = merged + float(w) * tau
    merged = merged.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([merged], lr=lr)
    norms_sq = [tau.pow(2).sum().clamp(min=1e-12) for tau in taus]

    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = torch.zeros((), device=device, dtype=torch.float32)
        for tau, norm_sq, w in zip(taus, norms_sq, weights):
            residual = merged - tau
            projection = residual @ tau.transpose(-1, -2)
            loss = loss + float(w) * (projection.pow(2).sum() / norm_sq)
        loss.backward()
        optimizer.step()

    return merged.detach().to(device="cpu", dtype=task_deltas[0].dtype)


def wudi_merge(
    base_sd: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
    task_vectors: Sequence[Dict[str, "torch.Tensor"]],  # type: ignore[name-defined]
    weights: Sequence[float],
    num_steps: int = WUDI_DEFAULT_STEPS,
    lr: float = WUDI_DEFAULT_LR,
    device: Optional[str] = None,
    progress_cb=None,
) -> Tuple[Dict[str, "torch.Tensor"], dict]:  # type: ignore[name-defined]
    """WUDI-Merging: interference-free merging of linear layers.

    Each linear-layer task delta is refined to minimise cross-task
    interference; non-linear tensors (embeddings, LayerNorm, biases) are merged
    by weighted averaging. The result is added to the base state dict. Requires
    no training data or held-out validation set.
    """
    torch = _lazy_torch()
    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    w_sum = float(sum(weights)) or 1.0
    norm_w = [w / w_sum for w in weights]

    keys = list(task_vectors[0].keys())
    total = len(keys)
    merged_delta: Dict[str, "torch.Tensor"] = {}  # type: ignore[name-defined]
    n_linear = 0
    n_averaged = 0

    for i, name in enumerate(keys):
        reference = task_vectors[0][name]
        deltas = [tv[name] if name in tv else torch.zeros_like(reference)
                  for tv in task_vectors]
        if _is_linear_weight(name, reference):
            merged_delta[name] = _wudi_optimize_layer(deltas, norm_w, num_steps, lr, dev)
            n_linear += 1
        else:
            agg = torch.zeros_like(reference)
            for w, d in zip(norm_w, deltas):
                agg = agg + w * d
            merged_delta[name] = agg
            n_averaged += 1
        if progress_cb is not None:
            progress_cb(i + 1, total)

    merged_sd = {k: base_sd[k] + merged_delta.get(k, 0) for k in base_sd}
    for k in merged_sd:
        merged_sd[k] = merged_sd[k].to(base_sd[k].dtype)

    stats = {
        "method": "wudi",
        "num_steps": int(num_steps),
        "lr": float(lr),
        "device": str(dev),
        "wudi_linear_layers": n_linear,
        "averaged_layers": n_averaged,
        "weights": list(norm_w),
    }
    return merged_sd, stats


# ---------------------------------------------------------------------------
# Saving merged checkpoint
# ---------------------------------------------------------------------------

def save_merged_checkpoint(
    base_path: str,
    state_dict: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
    output_dir: str,
    prefer_safetensors: bool = True,
) -> str:
    """Save merged weights + config + tokenizer (copied from base) to output_dir."""
    torch = _lazy_torch()
    transformers = _lazy_transformers()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save config + tokenizer by re-loading from base path
    cfg = transformers.AutoConfig.from_pretrained(_resolve_path(base_path))
    cfg.save_pretrained(str(out))
    try:
        tok = transformers.AutoTokenizer.from_pretrained(_resolve_path(base_path))
        tok.save_pretrained(str(out))
    except Exception as e:
        logger.warning(f"could not copy tokenizer from base ({e}); skipping")

    saved = False
    if prefer_safetensors:
        try:
            from safetensors.torch import save_file
            # `metadata={"format":"pt"}` is required by transformers >=4.30 to
            # load the file back via AutoModel.from_pretrained.
            # Also: contiguous + cloned tensors avoid "shared memory" errors when
            # the state dict contains tied weights (e.g. tie_word_embeddings).
            clean = {k: v.contiguous().clone() for k, v in state_dict.items()}
            save_file(clean, str(out / "model.safetensors"),
                      metadata={"format": "pt"})
            saved = True
        except ImportError:
            pass
    if not saved:
        torch.save(state_dict, str(out / "pytorch_model.bin"))
    return str(out)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _verdict_style(v: str) -> str:
    return {
        "COMPATIBLE": "[bold green]COMPATIBLE[/bold green]",
        "RISKY": "[bold yellow]RISKY[/bold yellow]",
        "INCOMPATIBLE": "[bold red]INCOMPATIBLE[/bold red]",
    }.get(v, v)


# ---------------------------------------------------------------------------
# CLI: top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, "-V", "--version")
@click.option("--verbose", is_flag=True, help="Verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """MergeSE - Post-hoc model merging for Software Engineering."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _configure_verbosity(verbose)


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

@cli.command("tasks")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def cmd_tasks(as_json: bool) -> None:
    """List SE classification tasks supported by MergeSE."""
    if as_json:
        click.echo(json.dumps([{
            "name": t.name,
            "display": t.display,
            "input_kind": t.input_kind,
            "num_labels": t.num_labels,
            "metric": t.metric,
            "benchmarks": list(t.benchmarks),
            "csv_columns": list(t.csv_columns),
            "description": t.description,
        } for t in all_tasks()], indent=2))
        return

    Console, Table, *_ = _lazy_rich()
    console = Console()
    table = Table(title="MergeSE - supported SE tasks", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Input")
    table.add_column("Metric")
    table.add_column("Benchmarks")
    table.add_column("Description")
    for t in all_tasks():
        table.add_row(
            t.name,
            t.input_kind,
            t.metric,
            ", ".join(t.benchmarks) or "-",
            t.description,
        )
    console.print(table)
    console.print(
        "\nMerging itself is task-agnostic: any two models that share a base "
        "encoder + tokenizer can be merged. Use [bold]--encoder-only[/bold] when "
        "the input models target different tasks (different classifier heads)."
    )


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@cli.command("inspect")
@click.argument("models", nargs=-1, required=True)
@click.option("--base", "base", default=None, help="Shared pre-trained base for task-vector computation.")
@click.option("--json-out", "json_out", type=click.Path(), default=None,
              help="If set, also write the report as JSON to this path.")
@click.pass_context
def cmd_inspect(ctx: click.Context, models: Tuple[str, ...], base: Optional[str],
                json_out: Optional[str]) -> None:
    """Analyse the compatibility of two or more checkpoints."""
    if len(models) < 2:
        raise click.UsageError("inspect requires at least 2 model paths.")

    Console, Table, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _lazy_rich()
    console = Console()

    loaded: List[LoadedModel] = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(),
                  console=console, transient=True) as p:
        for m in models:
            task = p.add_task(f"loading {m}", total=None)
            loaded.append(load_model(m))
            p.remove_task(task)

    # ---- Identification table ----
    id_table = Table(title="Checkpoint identification", show_lines=False)
    id_table.add_column("Model", no_wrap=False)
    id_table.add_column("model_type")
    id_table.add_column("hidden")
    id_table.add_column("layers")
    id_table.add_column("vocab_size")
    id_table.add_column("tokenizer_sig")
    id_table.add_column("architectures")
    for m in loaded:
        id_table.add_row(
            m.path,
            str(m.config.get("model_type", "?")),
            str(m.hidden_size or "?"),
            str(m.num_hidden_layers or "?"),
            str(m.tokenizer_vocab_size or m.config.get("vocab_size", "?")),
            str(m.tokenizer_signature or "-"),
            ", ".join(m.architectures) or "-",
        )
    console.print(id_table)

    # ---- Pairwise task-vector analysis ----
    pair_rows: List[dict] = []
    if base:
        base_m = load_model(base, with_tokenizer=False)
        keys = _shared_keys([base_m, *loaded])
        # Per-model task vectors only include keys whose shape matches the base -
        # so two models may have differently-sized deltas (e.g. one has a different
        # vocab and skips embeddings). Per-pair, we flatten over the intersection.
        deltas = [_compute_task_vector(m.state_dict, base_m.state_dict, keys) for m in loaded]

        tv_table = Table(title=f"Pairwise task-vector metrics (base: {base})")
        tv_table.add_column("Pair")
        tv_table.add_column("Cosine similarity", justify="right")
        tv_table.add_column("Sign agreement", justify="right")
        tv_table.add_column("‖Δ_a‖", justify="right")
        tv_table.add_column("‖Δ_b‖", justify="right")
        tv_table.add_column("Shared params", justify="right")

        for i in range(len(loaded)):
            for j in range(i + 1, len(loaded)):
                common = sorted(set(deltas[i].keys()) & set(deltas[j].keys()))
                if not common:
                    pair_rows.append({"a": loaded[i].path, "b": loaded[j].path,
                                      "cosine": 0.0, "sign_agreement": 0.0,
                                      "norm_a": 0.0, "norm_b": 0.0, "shared": 0})
                    tv_table.add_row(loaded[i].path + "  <->  " + loaded[j].path,
                                     "-", "-", "-", "-", "0")
                    continue
                flat_i = _flatten_delta({k: deltas[i][k] for k in common})
                flat_j = _flatten_delta({k: deltas[j][k] for k in common})
                cos = _cosine_similarity(flat_i, flat_j)
                sign = _sign_agreement(flat_i, flat_j)
                shared_n = int(flat_i.numel())
                row = {
                    "a": loaded[i].path,
                    "b": loaded[j].path,
                    "cosine": cos,
                    "sign_agreement": sign,
                    "norm_a": float(flat_i.norm()),
                    "norm_b": float(flat_j.norm()),
                    "shared_params": shared_n,
                }
                pair_rows.append(row)
                tv_table.add_row(
                    f"{loaded[i].path}  <->  {loaded[j].path}",
                    f"{cos:+.4f}",
                    f"{sign*100:.1f}%",
                    f"{row['norm_a']:.2f}",
                    f"{row['norm_b']:.2f}",
                    f"{shared_n:,}",
                )
        console.print(tv_table)

    verdict = _verdict(loaded)
    console.print(f"\nOverall verdict: {_verdict_style(verdict)}\n")

    if json_out:
        report = {
            "verdict": verdict,
            "models": [
                {
                    "path": m.path,
                    "model_type": m.config.get("model_type"),
                    "hidden_size": m.hidden_size,
                    "num_hidden_layers": m.num_hidden_layers,
                    "vocab_size": m.tokenizer_vocab_size or m.config.get("vocab_size"),
                    "tokenizer_signature": m.tokenizer_signature,
                    "architectures": m.architectures,
                }
                for m in loaded
            ],
            "pairs": pair_rows,
            "base": base,
        }
        Path(json_out).write_text(json.dumps(report, indent=2))
        console.print(f"Wrote JSON report -> {json_out}")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

@cli.command("merge")
@click.argument("models", nargs=-1, required=True)
@click.option("--base", required=True, help="Path to the shared pre-trained base checkpoint.")
@click.option("--method", type=click.Choice(["ties", "dare-ties", "wudi", "average"]),
              default="ties", show_default=True)
@click.option("--trim-percentile", type=float, default=20.0, show_default=True,
              help="TIES trim threshold (percentile of |Δ| zeroed).")
@click.option("--drop-rate", type=float, default=0.3, show_default=True,
              help="DARE drop rate p (only used for dare-ties).")
@click.option("--wudi-steps", type=int, default=WUDI_DEFAULT_STEPS, show_default=True,
              help="WUDI Adam steps per linear layer (only used for --method wudi).")
@click.option("--wudi-lr", type=float, default=WUDI_DEFAULT_LR, show_default=True,
              help="WUDI Adam learning rate (only used for --method wudi).")
@click.option("--device", default=None,
              help='Torch device for WUDI optimisation ("cpu"/"cuda"); default: auto-detect.')
@click.option("--weights", default=None,
              help='Comma-separated per-model weights (e.g. "1,1,0.5"). Default: equal.')
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output directory for merged checkpoint.")
@click.option("--seed", type=int, default=42, show_default=True, help="DARE RNG seed.")
@click.option("--encoder-only/--include-heads", default=None,
              help="Skip classifier heads in the merge (default: auto - skip when heads "
                   "differ in shape across input models).")
@click.option("--task", type=click.Choice([*task_names(), ""]), default="",
              help="Optional task hint (clone_detection, vulnerability_detection, "
                   "defect_prediction, ...). Used only to label artifacts.")
@click.pass_context
def cmd_merge(ctx: click.Context, models: Tuple[str, ...], base: str, method: str,
              trim_percentile: float, drop_rate: float, wudi_steps: int, wudi_lr: float,
              device: Optional[str], weights: Optional[str],
              output: str, seed: int, encoder_only: Optional[bool], task: str) -> None:
    """Merge two or more checkpoints into a single HuggingFace model."""
    if len(models) < 2:
        raise click.UsageError("merge requires at least 2 model paths.")
    _seed_everything(seed)

    Console, Table, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _lazy_rich()
    console = Console()

    if weights:
        try:
            w = [float(x) for x in weights.split(",")]
        except ValueError as e:
            raise click.UsageError(f"--weights must be comma-separated numbers: {e}")
        if len(w) != len(models):
            raise click.UsageError(f"--weights has {len(w)} entries but {len(models)} models given.")
    else:
        w = [1.0] * len(models)

    # ---- Load all models ----
    loaded: List[LoadedModel] = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(),
                  console=console, transient=True) as p:
        t = p.add_task(f"loading base ({base})", total=None)
        base_m = load_model(base, with_tokenizer=True)
        p.remove_task(t)
        for m in models:
            t = p.add_task(f"loading {m}", total=None)
            loaded.append(load_model(m, with_tokenizer=False))
            p.remove_task(t)

    # ---- Sanity check architectures ----
    arch_keys = [(m.config.get("model_type"), m.hidden_size, m.num_hidden_layers) for m in [base_m, *loaded]]
    if len(set(arch_keys)) > 1:
        raise click.UsageError(
            "Models do not share the same architecture signature (model_type / hidden_size / num_hidden_layers). "
            "Run `mergese inspect` first."
        )
    # Tokenizer warning
    base_sig = base_m.tokenizer_signature
    for m in loaded:
        if m.tokenizer_signature and base_sig and m.tokenizer_signature != base_sig:
            console.print(f"[yellow]warning:[/yellow] tokenizer signature mismatch for {m.path}")

    # ---- Decide encoder-only behaviour ----
    head_shapes = []
    for m in loaded:
        for k, v in m.state_dict.items():
            if _is_classifier_head(k) and (k.endswith(".weight") or k.endswith(".bias")):
                head_shapes.append((k, tuple(v.shape)))
                break
    head_shape_set = set(s for _, s in head_shapes)
    auto_encoder_only = len(head_shape_set) > 1
    if encoder_only is None:
        encoder_only = auto_encoder_only
        if auto_encoder_only:
            console.print(
                "[yellow]heads-differ:[/yellow] models have heterogeneous classifier "
                f"heads {sorted(head_shape_set)}; merging encoder only "
                "(use --include-heads to override)."
            )
    elif not encoder_only and auto_encoder_only:
        console.print(
            "[yellow]warning:[/yellow] --include-heads requested but heads differ in "
            "shape; non-matching head tensors will still be skipped per-tensor."
        )

    # ---- Compute task vectors over shared keys ----
    shared = _shared_keys([base_m, *loaded])
    deltas = [
        _compute_task_vector(m.state_dict, base_m.state_dict, shared, encoder_only=encoder_only)
        for m in loaded
    ]

    # ---- Run merge ----
    t0 = time.time()
    if method == "ties":
        merged_sd, stats = ties_merge(base_m.state_dict, deltas, w, trim_percentile)
    elif method == "dare-ties":
        merged_sd, stats = dare_ties_merge(base_m.state_dict, deltas, w, trim_percentile, drop_rate, seed)
    elif method == "wudi":
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TimeElapsedColumn(), console=console, transient=True) as p:
            wtask = p.add_task("WUDI optimisation", total=None)

            def _wudi_progress(done: int, total: int) -> None:
                p.update(wtask, total=total, completed=done)

            merged_sd, stats = wudi_merge(
                base_m.state_dict, deltas, w,
                num_steps=wudi_steps, lr=wudi_lr, device=device,
                progress_cb=_wudi_progress,
            )
    elif method == "average":
        merged_sd, stats = average_merge(base_m.state_dict, deltas, w)
    else:
        raise click.UsageError(f"unknown method: {method}")
    elapsed = time.time() - t0

    # ---- Save ----
    out_path = save_merged_checkpoint(base, merged_sd, output)
    stats["weights"] = list(w)
    stats["elapsed_sec"] = elapsed
    stats["output_path"] = out_path
    stats["num_parameters"] = sum(int(v.numel()) for v in merged_sd.values())
    stats["encoder_only"] = bool(encoder_only)
    stats["task"] = task or None
    stats["heterogeneous_heads"] = bool(auto_encoder_only)

    # ---- Pretty report ----
    table = Table(title="Merge complete")
    table.add_column("Field")
    table.add_column("Value")
    for k, v in stats.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.6f}")
        else:
            table.add_row(k, str(v))
    console.print(table)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _resolve_dataset(task: str, dataset: Optional[str], test_file: Optional[str]) -> str:
    """Locate the CSV file to evaluate against.

    A `dataset` arg can be either:
      * a path to a local CSV (used directly), or
      * a registered benchmark name (the user must still provide a local copy
        via `--test-file`, since MergeSE does not download benchmarks).
    """
    if test_file:
        return test_file
    if dataset and Path(dataset).exists():
        return dataset
    # Maybe a registered benchmark name?
    spec = get_task(task) if task else None
    if dataset and spec and dataset in spec.benchmarks:
        raise click.UsageError(
            f"'{dataset}' is a known benchmark for task '{task}', but MergeSE does not "
            f"download it. Pass --test-file with a local CSV "
            f"(columns: {','.join(spec.csv_columns) or 'code,label or code1,code2,label'})."
        )
    raise click.UsageError(
        f"task '{task}' requires --test-file (CSV with columns: "
        f"code,label or code1,code2,label)."
    )


@cli.command("evaluate")
@click.argument("model_path")
@click.option("--task", type=click.Choice(task_names()),
              default="clone_detection", show_default=True,
              help="Registered SE task (run `mergese tasks` to list all).")
@click.option("--dataset", default=None,
              help="Benchmark short name (e.g. bigclonebench, devign, defects4j) or CSV path.")
@click.option("--test-file", default=None, type=click.Path(exists=True),
              help="CSV with columns: code1,code2,label OR code,label.")
@click.option("--batch-size", default=32, show_default=True)
@click.option("--max-length", default=512, show_default=True)
@click.option("--device", default=None, help='"cpu", "cuda", or auto-detect (default).')
@click.option("--limit", default=0, show_default=True,
              help="Stop after N rows (0 = full file).")
@click.option("--metric", type=click.Choice(["auto", "binary", "macro"]),
              default="auto", show_default=True,
              help="Metric mode override (default: derived from task or labels).")
@click.option("--json-out", "json_out", type=click.Path(), default=None)
@click.pass_context
def cmd_evaluate(ctx: click.Context, model_path: str, task: str, dataset: Optional[str],
                 test_file: Optional[str], batch_size: int, max_length: int,
                 device: Optional[str], limit: int, metric: str,
                 json_out: Optional[str]) -> None:
    """Evaluate a model on a software-engineering benchmark."""
    import csv

    torch = _lazy_torch()
    transformers = _lazy_transformers()
    Console, Table, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn = _lazy_rich()
    console = Console()

    csv_path = _resolve_dataset(task, dataset, test_file)
    if not Path(csv_path).exists():
        raise click.UsageError(f"file not found: {csv_path}")

    task_spec = get_task(task)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = _resolve_path(model_path)

    # Tokenizer load with fallback: some fine-tune checkpoints ship only a
    # tokenizer.json saved by a newer tokenizers version. If that fails, fall
    # back to the model's _name_or_path / model_name (the HF base it was
    # initialised from), then to a few well-known SE encoders.
    def _try_tokenizer(p):
        try:
            return transformers.AutoTokenizer.from_pretrained(p)
        except Exception as e:
            logger.debug(f"tokenizer load from {p} failed: {e}")
            return None

    tok = _try_tokenizer(resolved_model_path)
    if tok is None:
        # 1. config.json _name_or_path
        cfg_path = Path(resolved_model_path) / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                cand = cfg.get("_name_or_path") or ""
                if cand and cand != resolved_model_path:
                    logger.info(f"local tokenizer failed; falling back to {cand}")
                    tok = _try_tokenizer(cand)
            except Exception:
                pass
        # 2. best_metrics.json model_name
        if tok is None:
            bm_path = Path(resolved_model_path) / "best_metrics.json"
            if bm_path.exists():
                try:
                    bm = json.loads(bm_path.read_text())
                    cand = bm.get("model_name")
                    if cand:
                        logger.info(f"tokenizer fallback via best_metrics.json: {cand}")
                        tok = _try_tokenizer(cand)
                except Exception:
                    pass
        # 3. Last-resort guess from model_type
        if tok is None:
            cfg_path = Path(resolved_model_path) / "config.json"
            if cfg_path.exists():
                try:
                    mt = json.loads(cfg_path.read_text()).get("model_type", "")
                    guess = {"roberta": "microsoft/codebert-base",
                             "bert": "bert-base-uncased"}.get(mt)
                    if guess:
                        logger.warning(f"using tokenizer from {guess} (model_type={mt})")
                        tok = _try_tokenizer(guess)
                except Exception:
                    pass
    if tok is None:
        raise click.UsageError(
            f"could not load a tokenizer for {model_path}. The checkpoint's "
            f"tokenizer.json may be from a newer `tokenizers` version. "
            f"Try upgrading: pip install -U tokenizers transformers"
        )

    # A checkpoint may store its encoder via `model.save_pretrained` and keep a
    # separate one-layer classifier in `classifier_head.bin` (a dict with keys
    # `classifier` / `dropout` / `num_labels`). When that file is present, wrap
    # the encoder with a matching head; otherwise use the standard HF path.
    custom_head_path = Path(resolved_model_path) / "classifier_head.bin"
    if custom_head_path.exists():
        logger.info("found classifier_head.bin - using custom-head evaluation path")
        head_sd = torch.load(str(custom_head_path), map_location="cpu", weights_only=False)
        num_labels = int(head_sd.get("num_labels", 2))
        dropout_p = float(head_sd.get("dropout", 0.1))
        encoder = transformers.AutoModel.from_pretrained(resolved_model_path)
        hidden = int(encoder.config.hidden_size)

        class CustomHeadModel(torch.nn.Module):
            """Encoder + dropout + Linear(hidden->num_labels). Returns an object
            with `.logits` so the rest of the eval loop is unchanged."""
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.dropout = torch.nn.Dropout(dropout_p)
                self.classifier = torch.nn.Linear(hidden, num_labels)
                self.classifier.load_state_dict(head_sd["classifier"])

            def forward(self, input_ids=None, attention_mask=None, **kw):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                logits = self.classifier(self.dropout(cls))
                return type("Out", (), {"logits": logits})()

        model = CustomHeadModel().to(dev)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            resolved_model_path
        ).to(dev)
    model.eval()

    # Inspect CSV header & decide pair vs. single mode
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
    detected = detect_input_kind(header)
    if task_spec and task_spec.input_kind not in ("auto", detected):
        console.print(
            f"[yellow]warning:[/yellow] task '{task}' expects {task_spec.input_kind} input "
            f"but CSV header suggests '{detected}' - going with the CSV header."
        )
    pair_mode = (detected == "pair")

    y_true: List[int] = []
    y_pred: List[int] = []

    def _iter_rows():
        with open(csv_path, "r", encoding="utf-8") as fp:
            r = csv.DictReader(fp)
            for i, row in enumerate(r):
                if limit and i >= limit:
                    break
                yield row

    rows_buffer: List[dict] = []
    t0 = time.time()
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TimeElapsedColumn(), console=console, transient=True) as p:
        task_p = p.add_task(f"evaluating on {Path(csv_path).name}", total=None)
        for row in _iter_rows():
            rows_buffer.append(row)
            if len(rows_buffer) >= batch_size:
                _run_batch(model, tok, rows_buffer, pair_mode, max_length, dev, y_true, y_pred)
                rows_buffer.clear()
                p.advance(task_p, batch_size)
        if rows_buffer:
            _run_batch(model, tok, rows_buffer, pair_mode, max_length, dev, y_true, y_pred)

    # Resolve metric mode: explicit flag > task spec > auto
    if metric == "auto":
        mode = pick_metric(task_spec, y_true)
    else:
        mode = metric
    metrics = _compute_metrics(y_true, y_pred, mode=mode)
    metrics["elapsed_sec"] = time.time() - t0
    metrics["device"] = dev
    metrics["n_examples"] = len(y_true)
    metrics["pair_mode"] = pair_mode
    metrics["dataset"] = csv_path
    metrics["task"] = task

    table = Table(title=f"Evaluation: {model_path}  ·  task={task}  ·  mode={metrics['mode']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k in ("accuracy", "precision", "recall", "f1", "n_examples",
              "num_classes", "elapsed_sec", "device"):
        if k not in metrics:
            continue
        v = metrics[k]
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    # Per-class table for multi-class evaluations
    if metrics.get("per_class"):
        pc = Table(title="Per-class metrics")
        pc.add_column("class")
        pc.add_column("precision", justify="right")
        pc.add_column("recall", justify="right")
        pc.add_column("f1", justify="right")
        for cls, m in metrics["per_class"].items():
            pc.add_row(cls, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}")
        console.print(pc)

    if json_out:
        Path(json_out).write_text(json.dumps(metrics, indent=2))
        console.print(f"Wrote JSON results -> {json_out}")


def _run_batch(model, tok, rows, pair_mode, max_length, device, y_true, y_pred):
    torch = _lazy_torch()
    if pair_mode:
        a = [r["code1"] for r in rows]
        b = [r["code2"] for r in rows]
        enc = tok(a, b, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    else:
        a = [r["code"] for r in rows]
        enc = tok(a, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    pred = out.logits.argmax(dim=-1).cpu().tolist()
    y_pred.extend(pred)
    y_true.extend(int(r["label"]) for r in rows)


def _compute_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                     mode: str = "auto") -> dict:
    """Compute classification metrics.

    mode = 'binary'  -> precision/recall/F1 on class 1 (binary tasks)
    mode = 'macro'   -> macro-averaged precision/recall/F1 over all classes
    mode = 'auto'    -> 'binary' if labels ⊆ {0,1} else 'macro'
    """
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mode": mode}
    if mode == "auto":
        mode = "binary" if set(y_true) | set(y_pred) <= {0, 1} else "macro"

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    if mode == "binary":
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mode": "binary"}

    # macro F1
    labels = sorted(set(y_true) | set(y_pred))
    precs, recs, f1s = [], [], []
    per_class = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
        per_class[str(c)] = {"precision": prec, "recall": rec, "f1": f1}
    n = max(len(labels), 1)
    return {
        "accuracy": acc,
        "precision": sum(precs) / n,
        "recall": sum(recs) / n,
        "f1": sum(f1s) / n,
        "mode": "macro",
        "num_classes": len(labels),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command("export")
@click.argument("model_path")
@click.option("--format", "fmt", type=click.Choice(["huggingface", "onnx", "torchscript"]),
              default="huggingface", show_default=True)
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--max-length", default=512, show_default=True,
              help="Sequence length used for tracing (onnx / torchscript only).")
@click.pass_context
def cmd_export(ctx: click.Context, model_path: str, fmt: str, output: str,
               max_length: int) -> None:
    """Export a merged model for deployment."""
    torch = _lazy_torch()
    transformers = _lazy_transformers()
    Console, Table, *_ = _lazy_rich()
    console = Console()

    resolved = _resolve_path(model_path)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "huggingface":
        out.mkdir(parents=True, exist_ok=True)
        # Try AutoModelForSequenceClassification first, else AutoModel
        try:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(resolved)
        except Exception:
            model = transformers.AutoModel.from_pretrained(resolved)
        tok = transformers.AutoTokenizer.from_pretrained(resolved)
        model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    elif fmt == "onnx":
        try:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(resolved)
        except Exception:
            model = transformers.AutoModel.from_pretrained(resolved)
        model.eval()
        tok = transformers.AutoTokenizer.from_pretrained(resolved)
        dummy = tok("def add(a,b): return a+b", return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_length)
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(out),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits":         {0: "batch"},
            },
            opset_version=14,
        )
        size = out.stat().st_size
    elif fmt == "torchscript":
        try:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                resolved, torchscript=True
            )
        except Exception:
            model = transformers.AutoModel.from_pretrained(resolved, torchscript=True)
        model.eval()
        tok = transformers.AutoTokenizer.from_pretrained(resolved)
        dummy = tok("def add(a,b): return a+b", return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_length)
        traced = torch.jit.trace(model, (dummy["input_ids"], dummy["attention_mask"]))
        traced.save(str(out))
        size = out.stat().st_size
    else:
        raise click.UsageError(f"unknown format: {fmt}")

    table = Table(title="Export complete")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("format", fmt)
    table.add_row("output", str(out))
    table.add_row("size_bytes", str(size))
    table.add_row("size_mb", f"{size / (1024*1024):.2f}")
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
