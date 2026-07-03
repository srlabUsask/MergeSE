# MergeSE Web Tool

A single-page web UI + REST API for the MergeSE merging engine. The server
is a thin Flask wrapper that shells out to the [CLI](CLI.md), so the
behaviour of every action is identical across both surfaces.

Features:

- **Checkpoint Library** - drag-and-drop a `.zip` of an HF folder once; every
  form (Inspect / Merge / Evaluate / Export) picks from the same library.
- **Async jobs with live logs** - every action returns a `job_id`; logs stream
  to the browser via Server-Sent Events.
- **Multi-source model references** - HuggingFace Hub IDs, your uploads,
  admin-mounted server checkpoints, or the output of a prior job (chainable).
- **Bundled benchmarks** - 200-row balanced samples of BigCloneBench,
  CLCDSA, and GPTCloneBench ship with the repo for smoke testing.
- **Downloadable artifacts** - every job exposes a one-click zip of its
  output directory.

---

## Installation

### Option A - Docker (recommended)

```bash
git clone https://github.com/srlabUsask/MergeSE.git
cd MergeSE

docker compose up -d --build
# Container listens on 127.0.0.1:8765
```

Open <http://localhost:8765>.

#### Common first-run issues

- **`permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`**
  - your user isn't in the `docker` group. One-time fix:
  ```bash
  sudo usermod -aG docker $USER
  # then open a fresh shell (close + reopen the terminal, or re-SSH)
  ```
  Until then, prefix `docker compose ...` with `sudo`.
- **`failed to bind host port 127.0.0.1:8765/tcp: address already in use`** -
  something else is bound to host port 8765 (a stale dev server, VS Code's
  port-forwarder, etc.). Either stop it
  (`sudo ss -ltnp | grep :8765` to find the PID), or remap the host port in
  `docker-compose.yml`:
  ```yaml
  ports:
    - "127.0.0.1:8766:8765"   # host 8766 -> container 8765
  ```
  Then `docker compose up -d` and use <http://localhost:8766>.

To put nginx + TLS in front (for example, at `mergese.example.com`):

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/mergese
sudo ln -s /etc/nginx/sites-available/mergese /etc/nginx/sites-enabled/
sudo certbot --nginx -d mergese.example.com
sudo systemctl reload nginx
```

The bundled `deploy/nginx.conf` already disables proxy buffering on
`/api/jobs/<id>/stream` (required for SSE live logs).

### Option B - bare-metal (Python venv)

```bash
git clone https://github.com/srlabUsask/MergeSE.git
cd MergeSE

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install ".[server,datasets]"
```

Then either:

```bash
# Development server (Flask built-in)
python server/app.py
# -> http://localhost:8765
```

```bash
# Production (gunicorn)
gunicorn -c deploy/gunicorn.conf.py server.app:app
```

For a long-running service, drop in the included systemd unit:

```bash
# Edit User=, paths, and MERGESE_CHECKPOINTS for your host first.
sudo cp deploy/mergese.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mergese
```

### Requirements

- Python 3.10+ (only for bare-metal install)
- Docker 24+ + docker compose v2 (only for Option A)
- PyTorch 2.0+, transformers 4.40+
- Flask 3, flask-cors, gunicorn
- huggingface/datasets (optional - only needed for `hf-dataset://` refs)

---

## Workflow

1. **Upload once** into the **Checkpoint Library** at the top of the page
   (drag-and-drop a `.zip` of any HF folder - anything with `config.json` +
   `model.safetensors` or `pytorch_model.bin`).
2. **Upload a CSV dataset (optional)** via the `+ Upload .csv` button - any
   classification task with `code,label` or `code1,code2,label` columns.
   The three bundled samples appear automatically as "Bundled benchmarks".
3. In every form below, **pick from the library**. A single dropdown lists
   uploads, server-mounted checkpoints, finished merge/export outputs, and
   HuggingFace Hub suggestions. You can also type any Hub ID directly. No
   re-uploading between Inspect -> Merge -> Evaluate.
4. **Download** the merged checkpoint from the Jobs panel when it's done.

### UI sections

| Section    | What it does                                                                       |
|------------|------------------------------------------------------------------------------------|
| Library    | Upload-once registry of every model and dataset visible to this server.            |
| Inspect    | Submit 2+ models, view live compatibility verdict + pairwise table.                |
| Merge      | Pick method / trim / drop rate / weights; download the merged zip when done.       |
| Evaluate   | Run the merged model on a CSV benchmark; F1 / precision / recall as a table.       |
| Export     | Repackage to HuggingFace / ONNX / TorchScript; downloadable from Jobs.             |
| Jobs       | Persistent log of every run with downloadable artifacts and live SSE logs.         |
| Docs       | Brief mathematical sketch of TIES / DARE-TIES.                                     |

### Model-reference forms

| Source              | Reference form                  | What it is                                                  |
|---------------------|---------------------------------|-------------------------------------------------------------|
| HuggingFace Hub     | `microsoft/codebert-base`       | Any public Hub ID; the server fetches it.                   |
| Upload              | `upload://<token>`              | A `.zip` you dropped into the library.                      |
| Server-mounted      | `server://<name>`               | Subdir of `MERGESE_CHECKPOINTS`.                            |
| Job output          | `job://<id>/merged`             | A finished merge/export from this session.                  |

### Dataset-reference forms

| Source              | Reference form                                          | What it is                                |
|---------------------|---------------------------------------------------------|-------------------------------------------|
| Bundled             | `bundled://bigclonebench`, `bundled://clcdsa`, ...      | 200-row samples shipped in `data/benchmarks/`. |
| Upload              | `dataset://<token>`                                     | A CSV uploaded via the UI or API.         |
| HuggingFace dataset | `hf-dataset://<id>#split=test#columns=...`              | Fetched via `datasets.load_dataset`.      |
| Server-mounted      | `server-dataset://<rel/path.csv>`                       | Subdir of `MERGESE_DATASETS`.             |

---

## REST API

All endpoints return JSON. `POST` endpoints return `{ job_id }` immediately
and run the work asynchronously; live logs stream from
`GET /api/jobs/<id>/stream` (`text/event-stream`).

```
GET    /api/health
GET    /api/tasks                    # registered SE tasks
GET    /api/presets                  # example workflows
GET    /api/library                  # unified list of models + datasets
GET    /api/checkpoints              # admin-mounted (only if MERGESE_CHECKPOINTS set)

POST   /api/uploads                  # multipart: file=<.zip>; returns { token, ref }
GET    /api/uploads
DELETE /api/uploads/<token>

POST   /api/datasets                 # multipart: file=<.csv|.zip>
GET    /api/datasets
DELETE /api/datasets/<token>

POST   /api/inspect                  { models: [str], base?: str }
POST   /api/merge                    { models, base, method, trim_percentile,
                                       drop_rate, weights?, seed, task?, encoder_only? }
POST   /api/evaluate                 { model, task, dataset?, test_file?,
                                       batch_size, max_length, limit, metric? }
POST   /api/export                   { model, format, max_length }

GET    /api/jobs
GET    /api/jobs/<id>
GET    /api/jobs/<id>/log
GET    /api/jobs/<id>/stream         # text/event-stream
GET    /api/jobs/<id>/result         # final JSON (when --json-out was set)
GET    /api/jobs/<id>/download       # zip of the job's output directory
POST   /api/jobs/<id>/cancel
```

### End-to-end with curl

```bash
# 1. Zip up two HF checkpoint folders, then upload them
zip -qr bcb.zip    codebert_bcb
zip -qr clcdsa.zip codebert_clcdsa
TOKEN1=$(curl -s -F "file=@bcb.zip"    http://localhost:8765/api/uploads | jq -r .token)
TOKEN2=$(curl -s -F "file=@clcdsa.zip" http://localhost:8765/api/uploads | jq -r .token)

# 2. Start a merge
JOB=$(curl -s -X POST -H 'content-type: application/json' \
  -d "{\"models\":[\"upload://$TOKEN1\",\"upload://$TOKEN2\"],
       \"base\":\"microsoft/codebert-base\",
       \"method\":\"ties\",\"trim_percentile\":20,\"weights\":\"0.5,0.5\"}" \
  http://localhost:8765/api/merge | jq -r .job_id)

# 3. Poll until done, then download the merged checkpoint
while [ "$(curl -s http://localhost:8765/api/jobs/$JOB | jq -r .status)" != "done" ]; do
    sleep 2
done
curl -o merged.zip http://localhost:8765/api/jobs/$JOB/download
```

---

## Configuration (environment variables)

| Variable                    | Default                  | Purpose                                                |
|-----------------------------|--------------------------|--------------------------------------------------------|
| `MERGESE_BIN`               | `python mergese.py`      | How the server invokes the CLI.                        |
| `MERGESE_CHECKPOINTS`       | unset                    | Directory exposed under `server://`.                   |
| `MERGESE_DATASETS`          | unset                    | Directory of admin-mounted CSVs (recursive scan).      |
| `MERGESE_BENCHMARKS`        | `./data/benchmarks`      | Bundled benchmark CSVs + `index.json`.                 |
| `MERGESE_ARTIFACTS`         | `./artifacts`            | Where job logs + merged outputs go.                    |
| `MERGESE_UPLOADS`           | `./uploads`              | Where uploaded checkpoints + datasets land.            |
| `MERGESE_MAX_CONCURRENT`    | `2`                      | Concurrent jobs (semaphore).                           |
| `MERGESE_MAX_UPLOAD_BYTES`  | `3221225472` (3 GB)      | Per-upload size cap.                                   |
| `MERGESE_ALLOW_LOCAL_PATHS` | `0`                      | Set `1` to accept raw filesystem paths in API calls.   |
| `MERGESE_PORT`              | `8765`                   | Bind port.                                             |
| `MERGESE_HOST`              | `0.0.0.0`                | Bind host.                                             |
| `HF_HOME`                   | `~/.cache/huggingface`   | Standard HuggingFace cache location.                   |

---

## Frontend behaviour

The frontend is plain HTML/CSS/JS (no build step). Same-origin by default.

- `?api=https://mergese.example.com` - point a locally-served frontend at a
  remote backend. CORS is enabled on the server.
- `?demo=1` - disables submit buttons; used by the GitHub Pages build for
  the read-only demo.
- Local filesystem paths in API requests are **rejected by default** -
  visitors must use uploads or HF Hub IDs. Set
  `MERGESE_ALLOW_LOCAL_PATHS=1` to relax this (trusted users only).

---

## Security notes

- The server runs jobs as subprocesses under its own user. Avoid running it
  as root.
- `MERGESE_ALLOW_LOCAL_PATHS=1` exposes the host filesystem to API callers
  - keep it off in any internet-facing deployment.
- Uploaded zips are extracted with path-traversal checks, but you should
  still place `MERGESE_UPLOADS` on a volume separate from the application
  code (the Docker compose file does this by default).
- For an internet-facing instance, run behind nginx + TLS using the
  bundled `deploy/nginx.conf`.
