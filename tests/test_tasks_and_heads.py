"""Tests for the task registry and head-mismatch handling.

These verify that MergeSE remains usable when merging models that target
different SE classification tasks (e.g. clone detection + vulnerability
detection), whose classifier heads have incompatible shapes.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mergese import (  # noqa: E402
    _compute_task_vector,
    _compute_metrics,
    _is_classifier_head,
    ties_merge,
    average_merge,
)
import mergese_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_registry_has_expected_tasks():
    names = set(mergese_tasks.names())
    for required in (
        "clone_detection", "vulnerability_detection", "defect_prediction",
        "code_smell_detection", "commit_classification", "code_review",
        "comment_classification", "type_inference", "exception_type", "custom",
    ):
        assert required in names, f"missing task: {required}"


def test_registry_specs_are_consistent():
    for t in mergese_tasks.all_tasks():
        assert t.input_kind in ("single", "pair", "auto")
        assert t.metric in ("binary_f1", "macro_f1", "accuracy", "auto")
        # pair tasks must have code1/code2 columns (except custom/auto)
        if t.input_kind == "pair" and t.csv_columns:
            assert "code1" in t.csv_columns and "code2" in t.csv_columns


def test_detect_input_kind():
    assert mergese_tasks.detect_input_kind(["code1", "code2", "label"]) == "pair"
    assert mergese_tasks.detect_input_kind(["code", "label"]) == "single"
    with pytest.raises(ValueError):
        mergese_tasks.detect_input_kind(["text", "label"])


def test_pick_metric_uses_task_then_labels():
    vuln = mergese_tasks.get("vulnerability_detection")
    commit = mergese_tasks.get("commit_classification")
    custom = mergese_tasks.get("custom")
    assert mergese_tasks.pick_metric(vuln,   [0, 1, 0, 1])     == "binary"
    assert mergese_tasks.pick_metric(commit, [0, 1, 2, 0])     == "macro"
    assert mergese_tasks.pick_metric(custom, [0, 1, 0, 1])     == "binary"
    assert mergese_tasks.pick_metric(custom, [0, 1, 2, 0])     == "macro"
    assert mergese_tasks.pick_metric(None,   [0, 0, 1, 1])     == "binary"


# ---------------------------------------------------------------------------
# head detection
# ---------------------------------------------------------------------------

def test_is_classifier_head_matches_common_patterns():
    assert _is_classifier_head("classifier.weight")
    assert _is_classifier_head("classifier.dense.weight")
    assert _is_classifier_head("classifier.out_proj.bias")
    assert _is_classifier_head("roberta.classifier.weight")
    assert _is_classifier_head("score.weight")
    assert _is_classifier_head("qa_outputs.weight")
    # Encoder tensors should not match
    assert not _is_classifier_head("encoder.layer.0.attention.self.query.weight")
    assert not _is_classifier_head("embeddings.word_embeddings.weight")


def test_compute_task_vector_skips_heads_when_encoder_only():
    base = {
        "encoder.layer.0.weight": torch.zeros(4, 4),
        "classifier.weight":      torch.zeros(2, 4),
    }
    model = {
        "encoder.layer.0.weight": torch.ones(4, 4),
        "classifier.weight":      torch.ones(2, 4),
    }
    delta_full = _compute_task_vector(model, base, base.keys(), encoder_only=False)
    delta_eo   = _compute_task_vector(model, base, base.keys(), encoder_only=True)
    assert "encoder.layer.0.weight" in delta_full and "classifier.weight" in delta_full
    assert "encoder.layer.0.weight" in delta_eo
    assert "classifier.weight" not in delta_eo


def test_compute_task_vector_drops_shape_mismatched_heads():
    """Even with encoder_only=False, shape-mismatched heads must be skipped silently."""
    base = {
        "encoder.weight":    torch.zeros(4, 4),
        "classifier.weight": torch.zeros(2, 4),   # binary head in base
    }
    model = {
        "encoder.weight":    torch.ones(4, 4),
        "classifier.weight": torch.ones(5, 4),    # 5-class head in fine-tuned model
    }
    delta = _compute_task_vector(model, base, base.keys(), encoder_only=False)
    assert "encoder.weight" in delta
    assert "classifier.weight" not in delta


def test_full_pipeline_with_mismatched_heads_uses_encoder_only_path():
    """End-to-end TIES merge: 2-class + 3-class models, encoder-only."""
    base = {
        "encoder.weight":    torch.zeros(8, 8),
        "classifier.weight": torch.zeros(2, 8),
    }
    sd_clone = {
        "encoder.weight":    torch.randn(8, 8),
        "classifier.weight": torch.randn(2, 8),
    }
    sd_commit = {
        "encoder.weight":    torch.randn(8, 8),
        "classifier.weight": torch.randn(3, 8),   # different shape!
    }
    d1 = _compute_task_vector(sd_clone,  base, base.keys(), encoder_only=True)
    d2 = _compute_task_vector(sd_commit, base, base.keys(), encoder_only=True)
    merged, stats = ties_merge(base, [d1, d2], [1.0, 1.0], 20.0)
    # base's head is preserved untouched (delta excluded it)
    assert torch.allclose(merged["classifier.weight"], base["classifier.weight"])
    # encoder is updated
    assert not torch.allclose(merged["encoder.weight"], base["encoder.weight"])


# ---------------------------------------------------------------------------
# multi-class metric
# ---------------------------------------------------------------------------

def test_metrics_binary_mode():
    m = _compute_metrics([0, 1, 1, 0, 1], [0, 1, 0, 0, 1], mode="binary")
    assert m["mode"] == "binary"
    assert m["accuracy"] == pytest.approx(4 / 5)
    # tp=2, fp=0, fn=1 -> prec=1, rec=2/3, f1=4/5
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"]    == pytest.approx(2 / 3)
    assert m["f1"]        == pytest.approx(0.8)


def test_metrics_macro_mode():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    m = _compute_metrics(y_true, y_pred, mode="macro")
    assert m["mode"] == "macro"
    assert m["num_classes"] == 3
    assert "per_class" in m and set(m["per_class"].keys()) == {"0", "1", "2"}
    # macro-F1 must lie within [0, 1]
    assert 0.0 <= m["f1"] <= 1.0


def test_metrics_auto_picks_binary_for_01():
    m = _compute_metrics([0, 1, 1, 0], [0, 1, 1, 1], mode="auto")
    assert m["mode"] == "binary"


def test_metrics_auto_picks_macro_for_multiclass():
    m = _compute_metrics([0, 1, 2, 0], [0, 2, 1, 0], mode="auto")
    assert m["mode"] == "macro"
