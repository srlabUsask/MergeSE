# MergeSE CLI

The `mergese` command-line tool merges fine-tuned HuggingFace encoder
checkpoints (CodeBERT, GraphCodeBERT, UniXcoder, CodeT5-encoder, ...) into a
single model **without any training data**, and evaluates the result on
standard software-engineering benchmarks.

It implements four task-vector-based merging algorithms:

- **Task-vector averaging** - simple mean of `θₖ - θ_base`.
- **TIES** (Yadav et al., NeurIPS 2023) - trim small deltas, elect a
  per-parameter majority sign, sum only sign-agreeing entries.
- **DARE-TIES** (Yu et al., 2024) - randomly drop a fraction `p` of delta
  entries and rescale by `1/(1-p)`, then apply TIES.
- **WUDI** (Cheng et al., ICML 2025) - optimise each linear-layer delta to
  minimise cross-task interference; average non-linear tensors (embeddings,
  LayerNorm, biases). Data-free, but the most compute-intensive method.

---

## Installation

### Option A - install as a package (recommended)

```bash
git clone https://github.com/srlabUsask/MergeSE.git
cd MergeSE

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU-only PyTorch wheel keeps the install small. Skip this line for the
# default CUDA wheel.
pip install --index-url https://download.pytorch.org/whl/cpu torch

pip install .
```

This exposes a `mergese` console script on your `$PATH`:

```bash
mergese --version
mergese --help
```

Add `[datasets]` to pull in `huggingface/datasets` for `hf-dataset://` refs:

```bash
pip install ".[datasets]"
```

### Option B - run directly from source

```bash
git clone https://github.com/srlabUsask/MergeSE.git
cd MergeSE
pip install -r requirements.txt
python mergese.py --help
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.40+
- safetensors, click, rich, tqdm, numpy

GPU is optional. CPU works for everything except large-checkpoint evaluation,
which becomes slow.

---

## Commands

```
mergese tasks      List supported SE tasks and benchmarks
mergese inspect    Analyse checkpoint compatibility
mergese merge      Merge checkpoints (TIES / DARE-TIES / WUDI / average)
mergese evaluate   Evaluate a model on any registered SE benchmark
mergese export     Export a merged model (HF / ONNX / TorchScript)
```

Every command supports `--verbose`. `inspect` and `evaluate` additionally
accept `--json-out report.json` for machine-readable output.

### `mergese tasks`

Lists the registered SE classification tasks, their input shape (single vs.
pair), default metric, and the benchmarks each task is known to be evaluated
on.

```bash
mergese tasks
mergese tasks --json
```

| Task                       | Input  | Metric    | Benchmarks                                    |
|----------------------------|--------|-----------|-----------------------------------------------|
| Clone detection            | pair   | binary F1 | BigCloneBench, CLCDSA, GPTCloneBench, POJ-104 |
| Vulnerability detection    | single | binary F1 | Devign, ReVeal, Big-Vul, D2A, Draper          |
| Defect / bug prediction    | single | binary F1 | Defects4J, PROMISE, CodeXGLUE-Defect          |
| Code-smell detection       | single | binary F1 | MLCQ, Qualitas                                |
| Commit classification      | single | macro F1  | CommitBench                                   |
| Code-review acceptability  | pair   | binary F1 | CodeReview                                    |
| Comment-code consistency   | pair   | binary F1 | comment-consistency datasets                  |
| Exception-type prediction  | single | macro F1  | CodeXGLUE-Exception                           |
| Type inference (closed)    | single | macro F1  | Typilus, Type4Py, ManyTypes4Py                |
| **Custom** (any CSV)       | auto   | auto      | -                                             |

### `mergese inspect`

Verify two or more checkpoints share a tokenizer, architecture, and base.
With `--base`, also reports pairwise task-vector cosine similarity and sign
agreement, plus a `COMPATIBLE / RISKY / INCOMPATIBLE` verdict.

```bash
mergese inspect \
    ./checkpoints/codebert_bcb \
    ./checkpoints/codebert_clcdsa \
    --base microsoft/codebert-base \
    --json-out inspect.json
```

Verdicts:

- `COMPATIBLE` - same tokenizer, same architecture, plausibly mergeable.
- `RISKY` - same tokenizer but different architecture/base detected; merge
  may still work but read the per-tensor warnings.
- `INCOMPATIBLE` - tokenizer mismatch; do not merge.

### `mergese merge`

TIES, DARE-TIES, WUDI, or average. Writes a full HuggingFace checkpoint (config
+ weights + tokenizer) to `--output`. Reports merge time plus per-method
statistics (fraction trimmed and sign-conflict fraction for TIES/DARE-TIES; the
number of WUDI-optimised linear layers for WUDI).

```bash
# TIES with equal weights, 20% trim
mergese merge \
    ./checkpoints/codebert_bcb \
    ./checkpoints/codebert_clcdsa \
    --base microsoft/codebert-base \
    --method ties \
    --output ./merged_ties

# DARE-TIES with three models and custom weights
mergese merge m1 m2 m3 \
    --base microsoft/codebert-base \
    --method dare-ties --drop-rate 0.3 --trim-percentile 20 \
    --weights 1,1,0.5 \
    --output ./merged_dt

# WUDI on a GPU (300 Adam steps per linear layer, lr 1e-5)
mergese merge \
    ./checkpoints/codebert_bcb \
    ./checkpoints/codebert_clcdsa \
    --base microsoft/codebert-base \
    --method wudi --wudi-steps 300 --wudi-lr 1e-5 --device cuda \
    --output ./merged_wudi
```

Key options:

| Flag                 | Default | Meaning                                                |
|----------------------|---------|--------------------------------------------------------|
| `--method`           | `ties`  | `ties` / `dare-ties` / `wudi` / `average`              |
| `--trim-percentile`  | `20.0`  | TIES trim, in percent (`0` keeps everything)           |
| `--drop-rate`        | `0.3`   | DARE drop fraction (only for `dare-ties`)              |
| `--wudi-steps`       | `300`   | WUDI Adam steps per linear layer (only for `wudi`)     |
| `--wudi-lr`          | `1e-5`  | WUDI Adam learning rate (only for `wudi`)              |
| `--device`           | auto    | Torch device for WUDI optimisation (`cpu` / `cuda`)    |
| `--weights`          | equal   | Comma-separated per-model λ values                     |
| `--seed`             | `42`    | RNG seed for DARE                                      |
| `--encoder-only`     | auto    | Force encoder-only merge (skip all heads)              |
| `--include-heads`    | off     | Force per-tensor merge of heads even if shapes differ  |
| `--task`             | none    | Hint used to size the merged classifier head           |

**WUDI note.** WUDI runs a short per-layer optimisation rather than a
closed-form combination, so it is the slowest method - expect roughly a minute
per merge on GPU and several minutes on CPU for a base-size encoder. Lower
`--wudi-steps` to trade quality for speed.

**Cross-task merging.** When models have differently-shaped classifier heads
(e.g. a 2-class clone detector + a 10-class commit classifier), MergeSE
auto-detects the mismatch and merges the encoder only. The base's head is
preserved, and you can attach a fresh task-specific head downstream:

```bash
mergese merge \
    ./checkpoints/codebert_bcb \
    ./checkpoints/codebert_devign \
    ./checkpoints/codebert_defects4j \
    --base microsoft/codebert-base \
    --method dare-ties --encoder-only \
    --output ./merged_universal
```

### `mergese evaluate`

Loads the model as `AutoModelForSequenceClassification` and runs on a CSV.
Reports accuracy, precision, recall, and F1. For multi-class tasks, reports
macro-F1 with a per-class breakdown.

CSV shape is auto-detected:

- `code,label`            - single-input classification
- `code1,code2,label`     - pair classification

```bash
mergese evaluate ./merged_ties \
    --task clone_detection \
    --test-file ./data/bcb_test.csv \
    --batch-size 64 \
    --metric binary
```

Useful flags:

| Flag             | Default | Meaning                                              |
|------------------|---------|------------------------------------------------------|
| `--task`         | -       | Registry name (`mergese tasks` to see options)       |
| `--test-file`    | -       | Path to a CSV (see column shapes above)              |
| `--batch-size`   | `32`    | Inference batch size                                 |
| `--max-length`   | `512`   | Tokenizer max length                                 |
| `--limit`        | `0`     | Cap the number of rows (`0` = full set)              |
| `--metric`       | `auto`  | `auto` / `binary` / `macro`                          |
| `--device`       | auto    | `cuda` / `cpu` / specific device id                  |
| `--json-out`     | none    | Write the full report as JSON                        |

### `mergese export`

Repackage a merged model as a HuggingFace folder, an ONNX graph (dynamic
batch + sequence axes), or a TorchScript module.

```bash
mergese export ./merged_ties --format huggingface --output ./merged_hf
mergese export ./merged_ties --format onnx        --output ./merged.onnx
mergese export ./merged_ties --format torchscript --output ./merged.ts
```

---

## End-to-end example

```bash
# 1. List supported tasks
mergese tasks

# 2. Check that two clone detectors are compatible
mergese inspect \
    ./checkpoints/codebert_bcb \
    ./checkpoints/codebert_clcdsa \
    --base microsoft/codebert-base

# 3. Merge them
mergese merge \
    ./checkpoints/codebert_bcb \
    ./checkpoints/codebert_clcdsa \
    --base microsoft/codebert-base \
    --method ties --trim-percentile 20 \
    --output ./merged

# 4. Score the merge on a held-out set
mergese evaluate ./merged \
    --task clone_detection \
    --test-file ./data/bcb_test.csv

# 5. Export for deployment
mergese export ./merged --format onnx --output ./merged.onnx
```

---

## Programmatic use

The merge primitives are importable:

```python
from mergese import load_model, ties_merge, dare_ties_merge, wudi_merge, average_merge
from mergese_tasks import get as get_task, all_tasks

print([t.name for t in all_tasks()])
```

See `mergese.py` and `mergese_tasks.py` for the full API surface.

---

## Citing

Tool paper:

```bibtex
@inproceedings{roy2026mergese,
  author    = {Palash R. Roy and Banani Roy and Chanchal K. Roy and Kevin A. Schneider},
  title     = {MergeSE: Post-hoc Model Merging for Software Engineering Tasks Without Retraining},
  booktitle = {Proc. ASE Tool Track},
  year      = {2026}
}
```

Research paper (the underlying methodology):

```bibtex
@inproceedings{roy2026unified,
  author    = {Palash R. Roy and Banani Roy and Chanchal K. Roy and Kevin A. Schneider},
  title     = {A Unified Model for Cross-Domain Clone Detection via Model Merging},
  booktitle = {Proc. ASE},
  year      = {2026}
}
```
