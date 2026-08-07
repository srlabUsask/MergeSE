# MergeSE - Post-hoc Model Merging for Software Engineering

[![CI](https://github.com/srlabUsask/MergeSE/actions/workflows/ci.yml/badge.svg)](https://github.com/srlabUsask/MergeSE/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21797312.svg)](https://doi.org/10.5281/zenodo.21797312)

> Tool artifact for the ASE 2026 Tool-Track submission
> **"MergeSE: Post-hoc Model Merging for Software Engineering Tasks Without Retraining"**
>
> The underlying model-merging approach comes from our research-track
> submission (under review):
> **"A Unified Model for Cross-Domain Clone Detection via Model Merging"**

MergeSE merges fine-tuned HuggingFace encoder checkpoints (CodeBERT,
GraphCodeBERT, UniXcoder, CodeT5-encoder, ...) into a single model **without
any training data**, and evaluates the result on standard software-engineering
benchmarks. It ships as a single-file **CLI** and a **web tool** that share
the same engine.

It implements:

- **TIES-Merging** (Yadav et al., NeurIPS 2023)
- **DARE-TIES** (Yu et al., 2024)
- **WUDI-Merging** (Cheng et al., ICML 2025)
- **PCB-Merging** (Du et al., NeurIPS 2024)
- **Task-vector averaging** (Ilharco et al., 2022)

Plus end-to-end evaluation across the full range of SE classification tasks
(clone detection, vulnerability detection, defect prediction, code-smell
detection, commit classification, code-review acceptability, comment-code
consistency, exception-type prediction, type inference, and any custom CSV)
and one-command export to HuggingFace / ONNX / TorchScript.

---

## Three ways to use MergeSE

Pick whichever fits your environment - all three sit on top of the same
merging engine, so results are identical.

| #  | Path                          | Best for                                                       |
|----|-------------------------------|----------------------------------------------------------------|
| 1  | **Web tool via Docker**       | Easiest setup. One command brings up the UI and REST API.      |
| 2  | **Web tool without Docker**   | Same UI, but you'd rather run Flask directly in a venv.        |
| 3  | **CLI tool**                  | Scripting, headless servers, reproducible runs, paper-grade evaluation. |

Full references: [docs/WEB.md](docs/WEB.md) and [docs/CLI.md](docs/CLI.md).
Deploying to a dedicated VM: [docs/DEPLOY.md](docs/DEPLOY.md).

---

## Quickstart

```bash
git clone https://github.com/srlabUsask/MergeSE.git
cd MergeSE
```

### 1. Web tool with Docker

```bash
docker compose up -d --build      # -> http://localhost:8765
```

Open <http://localhost:8765> in your browser. Stop with `docker compose down`.

**Common first-run issues:**

- **`permission denied ... /var/run/docker.sock`** - your user isn't in the
  `docker` group. One-time fix: `sudo usermod -aG docker $USER`, then open a
  fresh shell. Or prefix the one-off command with `sudo`.
- **`address already in use ... 8765`** - something else is bound to port 8765.
  Either stop it (`sudo ss -ltnp | grep :8765` to find the PID), or remap the
  host port in `docker-compose.yml` (e.g. `"127.0.0.1:8766:8765"`) and use
  <http://localhost:8766> instead.

### 2. Web tool without Docker (Flask in a venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install ".[server,datasets]"

python server/app.py               # -> http://localhost:8765
```

For production, swap `python server/app.py` for
`gunicorn -c deploy/gunicorn.conf.py server.app:app`.

### 3. CLI tool

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install .

mergese inspect ./model_a ./model_b --base microsoft/codebert-base
mergese merge   ./model_a ./model_b --base microsoft/codebert-base \
                --method ties --output ./merged
mergese evaluate ./merged --task clone_detection --test-file ./test.csv
mergese export   ./merged --format onnx --output ./merged.onnx
```

This installs a `mergese` console script on your `$PATH`. To run without
installing, use `python mergese.py ...` after `pip install -r requirements.txt`.

---

## Supported SE tasks

| Task                        | Input  | Metric    | Known benchmarks                              |
|-----------------------------|--------|-----------|-----------------------------------------------|
| Clone detection             | pair   | binary F1 | BigCloneBench, CLCDSA, GPTCloneBench, POJ-104 |
| Vulnerability detection     | single | binary F1 | Devign, ReVeal, Big-Vul, D2A, Draper          |
| Defect / bug prediction     | single | binary F1 | Defects4J, PROMISE, CodeXGLUE-Defect          |
| Code-smell detection        | single | binary F1 | MLCQ, Qualitas                                |
| Commit classification       | single | macro F1  | CommitBench                                   |
| Code-review acceptability   | pair   | binary F1 | CodeReview                                    |
| Comment-code consistency    | pair   | binary F1 | comment-consistency datasets                  |
| Exception-type prediction   | single | macro F1  | CodeXGLUE-Exception                           |
| Type inference (closed)     | single | macro F1  | Typilus, Type4Py, ManyTypes4Py                |
| **Custom** (any CSV)        | auto   | auto      | -                                             |

`mergese tasks` (CLI) or `GET /api/tasks` (web) returns the same list.

### Cross-task merging

When models have differently-shaped classifier heads (e.g. a 2-class clone
detector + a 10-class commit classifier), MergeSE auto-detects the mismatch
and runs an **encoder-only** merge. The base's head is preserved so you can
attach a fresh task-specific head downstream. Force this with
`--encoder-only`, or override with `--include-heads`.

---

## Merge methods

| Method       | Idea                                                                                           | Key options                          |
|--------------|------------------------------------------------------------------------------------------------|--------------------------------------|
| `average`    | Mean of the task vectors `θₖ - θ_base`.                                                         | `--weights`                          |
| `ties`       | Trim small deltas, elect a per-parameter majority sign, sum only sign-agreeing entries.        | `--trim-percentile`, `--weights`     |
| `dare-ties`  | Randomly drop a fraction of delta entries and rescale, then apply TIES.                         | `--drop-rate`, `--trim-percentile`   |
| `wudi`       | Optimise each linear-layer delta to minimise cross-task interference; average the rest.        | `--wudi-steps`, `--wudi-lr`, `--device` |
| `pcb`        | Score every parameter by intra-task significance × cross-task competition, drop all but the top-scoring fraction, then combine score-weighted. | `--pcb-ratio`, `--pcb-lambda`, `--pcb-scope` |

WUDI (Weight Disentanglement Interference minimisation) refines the merged
weight of every linear layer so that it interferes as little as possible with
each task's own update direction. Like the other methods it needs **no training
data**; it runs a short per-layer optimisation instead. It is the most compute-
intensive method - use `--device cuda` when a GPU is available.

```bash
mergese merge \
    ./checkpoints/codebert_bcb \
    ./checkpoints/codebert_clcdsa \
    --base microsoft/codebert-base \
    --method wudi --wudi-steps 300 --wudi-lr 1e-5 \
    --output ./merged_wudi
```

PCB (Parameter Competition Balancing) weighs two signals for every
`(task, parameter)` pair before deciding what survives the merge:

* **intra-balancing** - how much that parameter matters *within* its own task
  vector (squared magnitude, normalised per task, sharpened by `exp`);
* **inter-balancing** - whether the task pulls with or against the consensus of
  all the other tasks (`tanh` of the parameter against the cross-task sum).

Their product is the competition score. Only the top `--pcb-ratio` fraction of
scores is kept and the survivors are combined as a score-weighted average,
scaled by `--pcb-lambda`. Parameters that fight the consensus score negative
and are dropped rather than averaged away, which is what separates PCB from
TIES' majority-sign vote. Like TIES and DARE-TIES it is data-free and needs no
optimisation, so it runs at roughly TIES' speed.

`--pcb-scope global` (the default) ranks scores across the whole task vector,
matching the formulation in the paper. `--pcb-scope tensor` applies the ratio
per parameter tensor instead: lower peak memory, and it stops the embedding
table from consuming the entire keep-budget.

```bash
mergese merge \
    ./checkpoints/codebert_bcb \
    ./checkpoints/codebert_clcdsa \
    ./checkpoints/codebert_gptclonebench \
    --base microsoft/codebert-base \
    --method pcb --pcb-ratio 0.1 --pcb-lambda 1.0 \
    --output ./merged_pcb
```

---

## Repository layout

```
MergeSE/
├── mergese.py              # the entire CLI (single file)
├── mergese_tasks.py        # task registry
├── server/
│   ├── app.py              # Flask backend
│   ├── auth.py             # API keys, anon tokens, quotas (opt-in)
│   ├── manage_keys.py      # offline admin CLI for API keys
│   └── presets.json        # example workflows
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── favicon.svg
├── deploy/
│   ├── provision.sh        # one-shot VM provisioning script
│   ├── cloud-init.yaml     # first-boot user data for a fresh VM
│   ├── nginx.conf          # reverse-proxy site
│   ├── mergese.service     # systemd unit
│   └── gunicorn.conf.py
├── data/
│   └── benchmarks/         # 200-row bundled samples + index.json
├── tests/
│   ├── test_merge_math.py
│   ├── test_server_security.py
│   ├── test_auth.py
│   ├── test_tasks_and_heads.py
│   └── test_wudi.py
├── docs/
│   ├── CLI.md              # full CLI reference
│   ├── WEB.md              # full web-tool reference
│   ├── SECURITY.md         # hardening, auth, and go-live checklist
│   └── DEPLOY.md           # VM deployment runbook
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Bundled benchmark samples

| Name                       | Rows         | Task                                   | Source                                       |
|----------------------------|--------------|----------------------------------------|----------------------------------------------|
| `bundled://bigclonebench`  | 200 (100/100)| clone detection (Java)                 | CodeXGLUE / BigCloneBench                    |
| `bundled://clcdsa`         | 200 (100/100)| cross-language clones (Java<->Python)    | CLCDSA Source Codes                          |
| `bundled://gptclonebench`  | 200 (100/100)| semantic clones (Java)                 | GPTCloneBench standalone                     |

These are sampled from the original benchmarks for smoke-testing only. For
paper-grade numbers, point `--test-file` at the full dataset.

---

## Security & public deployment

The web server is safe to run locally as-is. Before exposing it to untrusted
users, read **[docs/SECURITY.md](docs/SECURITY.md)**. In brief, the tool already:

- accepts **safetensors-only** uploads and loads every checkpoint with
  `weights_only=True`, so uploaded model files cannot execute code;
- extracts archives with Zip-Slip and zip-bomb defenses;
- runs each merge in a **sandboxed worker** — stripped environment, job-private
  directory, resource limits, a hard timeout, and (where the host supports it)
  **no network** via an unprivileged namespace;
- builds every CLI invocation as an argument list (`shell=False`), so there is
  no command-injection path.

Optional, off by default (`MERGESE_REQUIRE_AUTH=1`): per-caller **API keys** and
short-lived **anonymous tokens** (Turnstile-gated), with daily/active-job quotas
and per-job ownership. Mint keys with `python server/manage_keys.py mint`. TLS,
HTTP rate limiting, and security headers are configured at the nginx/ingress
layer — see the deployment doc.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and PR guidance.
Issues labelled `good first issue` are self-contained; comment on
one to claim it before you start.

---

## Citing

If you use MergeSE itself, please cite the tool paper:

```bibtex
@inproceedings{Roy2026MergeSE,
  author    = {Roy, Palash R. and Roy, Banani and Schneider, Kevin A. and Roy, Chanchal K.},
  title     = {{MergeSE}: Post-Hoc Model Merging for Software Engineering Tasks without Retraining},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering},
  series    = {ASE '26},
  year      = {2026},
  month     = oct,
  location  = {Munich, Germany},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  isbn      = {979-8-4007-2882-2},
  numpages  = {5},
  doi       = {10.1145/3832783.3834630},
  url       = {https://doi.org/10.1145/3832783.3834630}
}
```

If you use the merging methodology in the MergeSE packages, please also cite our
research-track paper:

```bibtex
@inproceedings{Roy2026Unified,
  author    = {Roy, Palash R. and Roy, Banani and Schneider, Kevin A. and Roy, Chanchal K.},
  title     = {A Unified Model for Cross-Domain Clone Detection via Model Merging},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering},
  series    = {ASE '26},
  year      = {2026},
  month     = oct,
  location  = {Munich, Germany},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  numpages  = {13},
  doi       = {10.1145/3832783.3837415},
  url       = {https://doi.org/10.1145/3832783.3837415}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
