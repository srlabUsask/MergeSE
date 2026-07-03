"""
mergese_tasks
=============

Task registry for MergeSE.

A "task" here is a software-engineering classification problem that a
fine-tuned encoder can be evaluated on. The registry tells the rest of
MergeSE three things per task:

    1. Which CSV columns the test set uses (single-input vs. pair).
    2. Which named benchmarks belong to that task.
    3. How to score predictions (binary F1 vs. macro F1, # of classes).

The merging algorithms themselves are completely task-agnostic - they only
operate on state-dict tensors. The registry exists so that:

    * `evaluate` knows whether to expect `code` or `code1,code2` columns,
    * `inspect` can render dataset hints in the UI,
    * users can ask `mergese tasks` to discover what's supported,
    * downstream models with different classifier heads can be merged
      coherently (the merger uses the registry's `num_labels` hint when
      one is missing from the config).

Adding a new task is a one-liner - see the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskSpec:
    """Describes one software-engineering classification task."""
    name: str                       # canonical id (e.g. "clone_detection")
    display: str                    # human-readable label
    input_kind: str                 # "single" | "pair"
    num_labels: int                 # number of output classes
    metric: str                     # "binary_f1" | "macro_f1" | "accuracy"
    description: str
    benchmarks: Tuple[str, ...] = ()  # known benchmark short names
    csv_columns: Tuple[str, ...] = ()  # required CSV columns
    examples: Tuple[str, ...] = ()    # example datasets / checkpoints


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> None:
    _REGISTRY[spec.name] = spec


def get(name: str) -> Optional[TaskSpec]:
    return _REGISTRY.get(name)


def all_tasks() -> List[TaskSpec]:
    return list(_REGISTRY.values())


def names() -> List[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in SE classification tasks
# ---------------------------------------------------------------------------

register(TaskSpec(
    name="clone_detection",
    display="Code clone detection",
    input_kind="pair",
    num_labels=2,
    metric="binary_f1",
    description=(
        "Decide whether two code snippets are functionally equivalent clones. "
        "Type-1 through Type-4 clones; binary classification on the pair."
    ),
    benchmarks=("bigclonebench", "clcdsa", "gptclonebench", "poj-104"),
    csv_columns=("code1", "code2", "label"),
    examples=(
        "microsoft/codebert-base + BigCloneBench",
        "microsoft/unixcoder-base + CLCDSA",
    ),
))

register(TaskSpec(
    name="vulnerability_detection",
    display="Vulnerability detection",
    input_kind="single",
    num_labels=2,
    metric="binary_f1",
    description=(
        "Decide whether a function/snippet is vulnerable. Binary classification "
        "over single inputs, typically C/C++ functions."
    ),
    benchmarks=("devign", "reveal", "big-vul", "d2a", "draper"),
    csv_columns=("code", "label"),
    examples=(
        "microsoft/codebert-base + Devign",
        "microsoft/graphcodebert-base + ReVeal",
    ),
))

register(TaskSpec(
    name="defect_prediction",
    display="Defect / bug prediction",
    input_kind="single",
    num_labels=2,
    metric="binary_f1",
    description=(
        "Predict whether a method / file is defective. Often called bug "
        "prediction; binary classification."
    ),
    benchmarks=("defects4j", "promise", "bugs.jar", "codexglue-defect"),
    csv_columns=("code", "label"),
    examples=(
        "microsoft/codebert-base + Defects4J",
        "microsoft/unixcoder-base + Devign-defect",
    ),
))

register(TaskSpec(
    name="code_smell_detection",
    display="Code-smell detection",
    input_kind="single",
    num_labels=2,
    metric="binary_f1",
    description=(
        "Detect anti-patterns (god class, long method, feature envy, ...). "
        "Binary by default; multi-label setups should be split per-smell."
    ),
    benchmarks=("mlcq", "qualitas",),
    csv_columns=("code", "label"),
    examples=("microsoft/codebert-base + MLCQ",),
))

register(TaskSpec(
    name="commit_classification",
    display="Commit classification",
    input_kind="single",
    num_labels=3,
    metric="macro_f1",
    description=(
        "Classify commit messages or diffs (perfective / corrective / adaptive, "
        "or fix / feat / refactor). Multi-class."
    ),
    benchmarks=("commitbench", "codesearchnet-commit",),
    csv_columns=("code", "label"),
    examples=("microsoft/codebert-base + CommitBench",),
))

register(TaskSpec(
    name="code_review",
    display="Code-review acceptability",
    input_kind="pair",
    num_labels=2,
    metric="binary_f1",
    description=(
        "Given a code change + reviewer comment, predict acceptance. Pair "
        "input where code1=diff, code2=comment."
    ),
    benchmarks=("codereview", "codereview-new",),
    csv_columns=("code1", "code2", "label"),
    examples=("microsoft/codereviewer + CodeReview",),
))

register(TaskSpec(
    name="comment_classification",
    display="Code-comment classification",
    input_kind="pair",
    num_labels=2,
    metric="binary_f1",
    description=(
        "Decide whether a comment is consistent with the code it documents. "
        "Pair input: code1=comment, code2=code."
    ),
    benchmarks=("ccdetector", "datasets/comment-consistency"),
    csv_columns=("code1", "code2", "label"),
    examples=("microsoft/codebert-base + comment-consistency",),
))

register(TaskSpec(
    name="type_inference",
    display="Type inference (classification)",
    input_kind="single",
    num_labels=100,   # placeholder; usually closed-set top-K type vocab
    metric="macro_f1",
    description=(
        "Predict a variable's type from surrounding code context. Classification "
        "over a closed type vocabulary (override num_labels per-dataset)."
    ),
    benchmarks=("typilus", "type4py", "manytypes4py"),
    csv_columns=("code", "label"),
    examples=("microsoft/codebert-base + ManyTypes4Py",),
))

register(TaskSpec(
    name="exception_type",
    display="Exception-type prediction",
    input_kind="single",
    num_labels=20,
    metric="macro_f1",
    description=(
        "Predict the exception class a try/except block catches or a method "
        "raises. Multi-class over the project's exception vocabulary."
    ),
    benchmarks=("codexglue-exception",),
    csv_columns=("code", "label"),
    examples=("microsoft/codebert-base + CodeXGLUE exception",),
))

register(TaskSpec(
    name="custom",
    display="Custom CSV",
    input_kind="auto",  # decided by header
    num_labels=0,
    metric="auto",
    description=(
        "Any user-supplied CSV. Columns: `code,label` (single input) or "
        "`code1,code2,label` (pair). Metric: binary F1 if labels ∈ {0,1}, "
        "else macro F1."
    ),
    benchmarks=(),
    csv_columns=(),
    examples=(),
))


# ---------------------------------------------------------------------------
# Helpers used by the CLI
# ---------------------------------------------------------------------------

def detect_input_kind(csv_header: List[str]) -> str:
    """Return 'pair' if the header has code1+code2, 'single' if it has code."""
    s = set(csv_header)
    if {"code1", "code2"} <= s:
        return "pair"
    if "code" in s:
        return "single"
    raise ValueError(
        f"CSV header must include 'code' or ('code1','code2'); got {csv_header}"
    )


def pick_metric(task: Optional[TaskSpec], y_true) -> str:
    """Choose a metric mode: 'binary' or 'macro'."""
    if task is None or task.metric == "auto":
        return "binary" if set(y_true) <= {0, 1} else "macro"
    if task.metric == "binary_f1":
        return "binary"
    if task.metric == "macro_f1":
        return "macro"
    return "binary"
