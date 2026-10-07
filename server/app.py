"""
MergeSE web server
==================

A thin Flask wrapper that exposes the mergese CLI as a REST + Server-Sent-Events
API. Jobs run as background subprocesses with streaming logs, so long-running
merges/evaluations don't block the request thread. Designed to be reverse-proxied
behind nginx at https://mergeSE.usask.ca.

Endpoints
---------
GET    /                            - static frontend (index.html)
GET    /api/health                  - { ok, version }
GET    /api/tasks                   - registered SE tasks (mirrors `mergese tasks --json`)
GET    /api/presets                 - example workflows
GET    /api/checkpoints             - env-gated list of admin-mounted checkpoints
POST   /api/uploads                 - upload a checkpoint as a .zip or files
GET    /api/uploads                 - list uploaded checkpoints
DELETE /api/uploads/<token>         - delete an upload

POST   /api/inspect|merge|evaluate|export
                                    - start a job; returns { job_id }
GET    /api/jobs                    - list all jobs
GET    /api/jobs/<id>               - job metadata
GET    /api/jobs/<id>/stream        - SSE log stream
GET    /api/jobs/<id>/result        - final JSON report (if --json-out was set)
GET    /api/jobs/<id>/log           - full log as text
POST   /api/jobs/<id>/cancel        - SIGTERM the subprocess
GET    /api/jobs/<id>/download      - stream the job's output directory as .zip

The web server NEVER auto-loads or downloads models - it only shells out to the
CLI binary, which is the single source of truth for behaviour.

Model references in API calls can be one of:
    * a HuggingFace Hub ID (e.g. "microsoft/codebert-base") - default
    * "upload://<token>"  - the server resolves this to the upload directory
    * "server://<name>"   - admin-mounted checkpoint under MERGESE_CHECKPOINTS
    * a plain absolute path - only honoured when MERGESE_ALLOW_LOCAL_PATHS=1
"""

from __future__ import annotations

import json
import os
import re
import shlex
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, Response, abort, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _first_existing_dir(candidates: List[Path], expect_file: Optional[str] = None) -> Path:
    """Return the first candidate directory that exists (and, when
    `expect_file` is given, contains that file). Falls back to the first
    candidate when nothing matches - keeps dev-tree behaviour unchanged while
    letting a wheel install resolve data from `sys.prefix/share/mergese/...`."""
    for c in candidates:
        if c.is_dir() and (expect_file is None or (c / expect_file).exists()):
            return c
    return candidates[0]


# Candidate layouts, in search order:
#   1. repo tree  (dev)                 - ROOT/frontend, ROOT/data/benchmarks
#   2. wheel install  (`pip install mergese`) - sys.prefix/share/mergese/...
#   3. user install  (`pip install --user`)   - site.USER_BASE/share/mergese/...
# `data-files` entries in pyproject.toml deposit files at option 2/3.
try:
    import site
    _USER_BASE = Path(site.getuserbase())
except Exception:
    _USER_BASE = Path.home()
_PREFIX = Path(sys.prefix)
FRONTEND = _first_existing_dir(
    [
        ROOT / "frontend",
        _PREFIX / "share" / "mergese" / "frontend",
        _USER_BASE / "share" / "mergese" / "frontend",
    ],
    expect_file="index.html",
)

# Ensure sibling modules (auth.py) import cleanly no matter how app.py is
# launched - `python server/app.py`, gunicorn, or importlib-by-path in tests.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ---- config from env ---------------------------------------------------------

# Default MERGESE_BIN: prefer the repo copy if present, else `python -m mergese`
# so a wheel install (where mergese.py isn't under ROOT) still works.
_default_bin = (
    sys.executable + " " + str(ROOT / "mergese.py")
    if (ROOT / "mergese.py").exists()
    else sys.executable + " -m mergese"
)
MERGESE_BIN = os.environ.get("MERGESE_BIN", _default_bin)
CHECKPOINTS_ROOT = os.environ.get("MERGESE_CHECKPOINTS", "")
DATASETS_ROOT = os.environ.get("MERGESE_DATASETS", "")  # optional admin-mounted CSVs
# B05: the job registry is in-process memory. Running multiple gunicorn
# workers makes job lookups 404 from any worker that didn't create the job.
# Loud warning + a visible banner in /api/health when the operator sets a
# worker count above 1 anywhere we can see it.
try:
    _mergese_workers = int(os.environ.get("MERGESE_WORKERS", "1"))
except (TypeError, ValueError):
    _mergese_workers = 1
if _mergese_workers > 1:
    import logging as _logging
    _logging.getLogger("mergese").warning(
        "MERGESE_WORKERS=%s: job state is in-process memory, so lookups "
        "will 404 unpredictably across workers. Set workers back to 1 "
        "and scale with --threads instead until job state moves to shared "
        "storage.", _mergese_workers,
    )


BENCHMARKS_ROOT = Path(os.environ.get(
    "MERGESE_BENCHMARKS",
    str(_first_existing_dir(
        [
            ROOT / "data" / "benchmarks",
            _PREFIX / "share" / "mergese" / "data" / "benchmarks",
            _USER_BASE / "share" / "mergese" / "data" / "benchmarks",
        ],
        expect_file="index.json",
    )),
))
ARTIFACTS_ROOT = Path(os.environ.get("MERGESE_ARTIFACTS", str(ROOT / "artifacts")))
UPLOADS_ROOT = Path(os.environ.get("MERGESE_UPLOADS", str(ROOT / "uploads")))
DATASET_UPLOADS_ROOT = UPLOADS_ROOT / "_datasets"
MAX_CONCURRENT = int(os.environ.get("MERGESE_MAX_CONCURRENT", "2"))
# Cap on simultaneously pending or running jobs. Submissions beyond this get a
# 429 so a flood of long-running merges cannot pile up without bound.
MAX_QUEUE = int(os.environ.get("MERGESE_MAX_QUEUE", str(MAX_CONCURRENT * 4)))
# Default upload cap: 3 GB (a typical RoBERTa-base checkpoint is ~500 MB; allow
# 6× headroom for larger encoders / zipped multi-file checkpoints).
MAX_UPLOAD_BYTES = int(os.environ.get("MERGESE_MAX_UPLOAD_BYTES", str(3 * 1024 ** 3)))
ALLOW_LOCAL_PATHS = bool(int(os.environ.get("MERGESE_ALLOW_LOCAL_PATHS", "0")))
# Uploaded checkpoints are deserialized by the merge worker. PyTorch `.bin`
# checkpoints are Python pickles and can execute arbitrary code on load, so by
# default we accept only tensor-only safetensors weights and refuse pickle
# formats outright. An operator who fully trusts their users can flip this, but
# it should stay off for any publicly reachable deployment.
ALLOW_PICKLE_UPLOADS = bool(int(os.environ.get("MERGESE_ALLOW_PICKLE_UPLOADS", "0")))
# Zip-bomb defenses for uploaded archives: cap the total uncompressed size, the
# number of entries, and the per-entry compression ratio. Defaults give ample
# room for a sharded multi-file encoder while refusing pathological archives.
MAX_UNCOMPRESSED_BYTES = int(os.environ.get(
    "MERGESE_MAX_UNCOMPRESSED_BYTES", str(12 * 1024 ** 3)))
MAX_ARCHIVE_ENTRIES = int(os.environ.get("MERGESE_MAX_ARCHIVE_ENTRIES", "10000"))
MAX_COMPRESSION_RATIO = float(os.environ.get("MERGESE_MAX_COMPRESSION_RATIO", "200"))
# File extensions that carry executable pickle payloads or code. Refused on
# upload unless ALLOW_PICKLE_UPLOADS is set.
_PICKLE_EXTS = {".bin", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".npy", ".npz",
                ".joblib", ".dill", ".model", ".h5", ".msgpack"}
_CODE_EXTS = {".py", ".pyc", ".pyo", ".so", ".sh", ".pyd", ".dll", ".dylib"}

# ---- worker sandbox config ---------------------------------------------------
# Each job runs in a hardened child process: a stripped environment, a
# job-private HOME/TMPDIR, POSIX rlimits, a hard wall-clock timeout, and - when
# the host supports an unprivileged network namespace - no network at all.
WORKER_TIMEOUT_SEC = int(os.environ.get("MERGESE_WORKER_TIMEOUT_SEC", "3600"))
WORKER_CPU_SEC = int(os.environ.get("MERGESE_WORKER_CPU_SEC", "3600"))
# Max bytes any single file the job writes may reach (RLIMIT_FSIZE). Guards
# against a job filling the disk with one enormous output. 20 GB default.
WORKER_MAX_FILE_BYTES = int(os.environ.get(
    "MERGESE_WORKER_MAX_FILE_BYTES", str(20 * 1024 ** 3)))
# Process/thread cap (RLIMIT_NPROC). Off by default (0): on Linux this counts
# EVERY process owned by the real uid, not just this job, so on a shared host a
# busy user trips it and torch fails with "can't start new thread". Reliable
# per-job PID limiting needs cgroups (pids.max), which requires privilege we do
# not assume here. Set a value only on a single-tenant box.
WORKER_MAX_NPROC = int(os.environ.get("MERGESE_WORKER_MAX_NPROC", "0"))
# Virtual-address-space cap (RLIMIT_AS) in bytes. 0 disables it: CUDA/torch
# reserve enormous virtual ranges, so an AS cap causes spurious OOM on GPU
# hosts. Leave at 0 unless you are on a CPU-only box and want a hard ceiling.
WORKER_AS_BYTES = int(os.environ.get("MERGESE_WORKER_AS_BYTES", "0"))
# Run the merge worker with no network. Default on: a merge of uploaded or
# already-cached checkpoints needs no network, and denying it removes the whole
# SSRF / network-probing surface. Turn off only if you accept on-demand Hub
# fetches from inside the worker.
WORKER_OFFLINE = bool(int(os.environ.get("MERGESE_WORKER_OFFLINE", "1")))
_UNSHARE_BIN = shutil.which("unshare")


def _probe_netns() -> bool:
    """True if this host can create an unprivileged network namespace.

    We shell out once at startup rather than trusting a config flag, because a
    namespace we cannot actually create would make every job fail to launch.
    """
    if not (_UNSHARE_BIN and WORKER_OFFLINE):
        return False
    try:
        r = subprocess.run([_UNSHARE_BIN, "-rn", "--", "true"],
                           stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


NETNS_OK = _probe_netns()
# Time-to-live for uploaded checkpoints, uploaded datasets, and job artifacts.
# A background janitor removes anything older so the disk cannot be filled.
UPLOAD_TTL_HOURS = float(os.environ.get("MERGESE_UPLOAD_TTL_HOURS", "48"))
ARTIFACT_TTL_HOURS = float(os.environ.get("MERGESE_ARTIFACT_TTL_HOURS", "48"))
JANITOR_INTERVAL_SEC = int(os.environ.get("MERGESE_JANITOR_INTERVAL", "3600"))
JANITOR_ENABLED = bool(int(os.environ.get("MERGESE_JANITOR", "1")))
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)


# ---- auth config -------------------------------------------------------------
# All OFF by default so a trusted single-tenant deployment is unchanged. For a
# public deployment set MERGESE_REQUIRE_AUTH=1 (and a Turnstile secret if you
# want anonymous website access).
REQUIRE_AUTH = bool(int(os.environ.get("MERGESE_REQUIRE_AUTH", "0")))
# Emergency switch: refuse anonymous tokens and require an API key, without a
# redeploy. Set MERGESE_DISABLE_ANON=1 if the public endpoint is being abused.
DISABLE_ANON = bool(int(os.environ.get("MERGESE_DISABLE_ANON", "0")))
ADMIN_TOKEN = os.environ.get("MERGESE_ADMIN_TOKEN", "")
TURNSTILE_SECRET = os.environ.get("MERGESE_TURNSTILE_SECRET", "")
# Public Turnstile site key - safe to expose to the browser. The front-end
# fetches it from /api/v1/config to render the challenge widget.
TURNSTILE_SITE_KEY = os.environ.get("MERGESE_TURNSTILE_SITE_KEY", "")
AUTH_DB = Path(os.environ.get("MERGESE_AUTH_DB", str(ARTIFACTS_ROOT / "_auth" / "auth.db")))
# 24 hours by default so a visitor who uploads a model and comes back after
# lunch doesn't lose access to their own uploads. Still bounded for a shared
# anonymous identity namespace. Operators tune with MERGESE_ANON_TTL_SEC.
ANON_TTL_SEC = int(os.environ.get("MERGESE_ANON_TTL_SEC", "86400"))

_AUTH = None  # lazily created AuthStore


def _auth_store():
    global _AUTH
    if _AUTH is None:
        import auth as _authmod
        secret = _authmod.load_secret(
            os.environ.get("MERGESE_AUTH_SECRET"),
            AUTH_DB.parent / "signing.secret")
        _AUTH = _authmod.AuthStore(AUTH_DB, secret, anon_ttl_sec=ANON_TTL_SEC)
    return _AUTH


def _load_benchmarks_index() -> dict:
    """Read data/benchmarks/index.json (the bundled-samples catalogue)."""
    p = BENCHMARKS_ROOT / "index.json"
    if not p.exists():
        return {"benchmarks": []}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"benchmarks": []}


# ---- job model ---------------------------------------------------------------

@dataclass
class Job:
    id: str
    cmd: List[str]
    kind: str  # inspect / merge / evaluate / export
    status: str = "pending"  # pending / running / done / error / cancelled
    pid: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    log_path: Path = field(default_factory=Path)
    result_path: Optional[Path] = None
    error: Optional[str] = None
    # True when the job must reach the network (e.g. an uncached HuggingFace
    # Hub id). Such a job cannot run in the offline namespace; it is refused
    # unless the operator has explicitly allowed a networked worker.
    needs_network: bool = False
    # Identity that submitted the job (None when auth is disabled). Used to keep
    # one caller from reading, cancelling, or downloading another's job.
    owner: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "pid": self.pid,
            "cmd": " ".join(shlex.quote(c) for c in self.cmd),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "params": self.params,
            "artifacts": self.artifacts,
            "error": self.error,
        }


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
JOB_PROCS: Dict[str, subprocess.Popen] = {}
RUN_SEMA = threading.BoundedSemaphore(MAX_CONCURRENT)


def _inflight_jobs() -> int:
    """Number of jobs currently pending or running."""
    with JOBS_LOCK:
        return sum(1 for j in JOBS.values() if j.status in ("pending", "running"))


def _capacity_response():
    """Return a 429 response tuple when the job queue is saturated, else None."""
    if _inflight_jobs() >= MAX_QUEUE:
        return jsonify({
            "error": "server is at capacity; too many jobs are queued or running",
            "max_queue": MAX_QUEUE,
            "retry_after_sec": 30,
        }), 429
    return None


# ---- authentication middleware ----------------------------------------------

def _bearer_token() -> Optional[str]:
    h = request.headers.get("Authorization", "")
    if h.startswith("Bearer "):
        return h[len("Bearer "):].strip()
    return None


def _request_token() -> Optional[str]:
    """Caller token from the Bearer header, X-Anon-Token header, or ?token= query.

    The query-param form exists for EventSource (SSE), which cannot set custom
    request headers - the job-log stream passes its token that way.
    """
    t = _bearer_token()
    if t:
        return t
    x = request.headers.get("X-Anon-Token")
    if x:
        return x.strip()
    q = request.args.get("token")
    if q:
        return q.strip()
    return None


def _resolve_client():
    """Resolve the calling identity for this request.

    Returns an auth.Client, or None when auth is disabled (trusted single-tenant
    mode). Raises auth.AuthError when auth is required and the caller is not
    valid; the registered error handler turns that into a JSON response.
    """
    if not REQUIRE_AUTH:
        return None
    import auth as _authmod
    token = _request_token()
    if not token:
        raise _authmod.AuthError(401, "authentication required")
    api_key = token if token.startswith(_authmod.KEY_PREFIX) else None
    anon = None if api_key else token
    if anon and DISABLE_ANON:
        raise _authmod.AuthError(
            403, "anonymous access is temporarily disabled; use an API key")
    return _auth_store().authenticate(api_key, None if DISABLE_ANON else anon)


def _client_active_jobs(client_id: str) -> int:
    with JOBS_LOCK:
        return sum(1 for j in JOBS.values()
                   if j.owner == client_id and j.status in ("pending", "running"))


def _authenticate() -> Optional[str]:
    """Authenticate the request and return the caller's client id.

    Returns None when auth is disabled (trusted single-tenant mode). Raises
    auth.AuthError 401 when auth is on and the token is bad/missing. Does NOT
    reserve quota; use `_reserve_job_quota` for that once refs are validated,
    so a mistyped model ref doesn't eat the caller's daily budget.
    """
    client = _resolve_client()
    return client.client_id if client else None


def _reserve_job_quota() -> None:
    """Reserve one job against the authenticated caller's daily budget.

    No-op when auth is disabled. Raises auth.AuthError 429 when the quota is
    exceeded. Call this AFTER `_authenticate` and AFTER the refs have been
    validated, so a 400 for a bad ref doesn't consume the caller's budget.
    """
    client = _resolve_client()
    if client is None:
        return
    _auth_store().check_and_reserve(client, _client_active_jobs(client.client_id))


def _authorize_job_submission() -> Optional[str]:
    """Back-compat wrapper: authenticate + reserve in one call. New endpoint
    code should call `_authenticate` and `_reserve_job_quota` separately so a
    bad-ref 400 doesn't consume quota."""
    owner = _authenticate()
    _reserve_job_quota()
    return owner


def _require_owner(job: "Job") -> None:
    """404 unless the current caller owns `job` (no-op when auth is off).

    Returns 404 rather than 403 so the existence of another caller's job is not
    revealed.
    """
    if not REQUIRE_AUTH:
        return
    client = _resolve_client()
    if client is None or job.owner != client.client_id:
        abort(404)


# ---- ownership for uploads / datasets / job artifacts ------------------------
#
# Public deployments share one filesystem across every anonymous visitor, so
# each upload / dataset / job dir gets a plain `.owner` file holding the
# client_id that created it. Listings filter by this value and resolvers refuse
# to turn a `upload://` / `dataset://` / `job://` ref into a path for anyone
# else's artifact. Dirs with no `.owner` file are treated as unowned and are
# only visible when auth is disabled (dev / trusted mode).

def _write_owner(dir_path: Path, owner: Optional[str]) -> None:
    """Record `owner` as the creator of this artifact dir. Silent on IO error:
    a missing .owner file simply makes the artifact inaccessible to any
    authenticated caller, which is the safe default."""
    if not owner:
        return
    try:
        (dir_path / ".owner").write_text(owner)
    except OSError:
        pass


def _read_owner(dir_path: Path) -> Optional[str]:
    try:
        txt = (dir_path / ".owner").read_text().strip()
        return txt or None
    except OSError:
        return None


def _owns(dir_path: Path, owner: Optional[str]) -> bool:
    """True if `owner` may use this upload/dataset/job dir. When auth is off
    (owner is None), everything is accessible, matching single-tenant mode."""
    if owner is None:
        return True
    return _read_owner(dir_path) == owner


# ---- model-reference resolution ---------------------------------------------

# A single HF Hub path component (org or model name). Must start with an
# alphanumeric/underscore/hyphen (NOT a dot) and may contain dots afterwards.
# Disallowing a leading dot stops `.` / `..` / `./secret` from passing as a
# "valid id" and later resolving to a relative path on disk (B03 retest).
_HF_COMPONENT = r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,95}"
_HF_ID = re.compile(rf"^{_HF_COMPONENT}(?:/{_HF_COMPONENT})?$")


def _json_body() -> dict:
    """Parse and validate a request body as a JSON object.

    Raises werkzeug's `BadRequest` (-> 400 with a readable error) when the body
    isn't valid JSON OR when it is valid JSON but not an object (e.g. a list,
    a string, a number). Without this check, a `request.get_json(force=True) or {}`
    on a list body sailed through and the next `body.get(...)` call raised
    AttributeError, which Flask converted to an opaque 500.
    """
    body = request.get_json(force=True, silent=True)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        abort(400, description="request body must be a JSON object")
    return body


# ---- per-field body validators ----------------------------------------------
# Catch bad field types (e.g. {"models": 42}) with a readable 400 instead of
# letting the handler reach a `len(42)` and 500. Keep validation narrow -
# only shapes the handlers actually read.

def _field_str(body: dict, key: str, *, required: bool = False,
               default: Optional[str] = None) -> Optional[str]:
    v = body.get(key, default)
    if v is None:
        if required:
            abort(400, description=f"missing required field {key!r}")
        return default
    if not isinstance(v, str):
        abort(400, description=f"{key!r} must be a string, got {type(v).__name__}")
    return v


def _field_str_list(body: dict, key: str, *, required: bool = False,
                    min_items: int = 0) -> List[str]:
    v = body.get(key)
    if v is None:
        if required or min_items > 0:
            abort(400, description=f"missing required field {key!r}")
        return []
    if not isinstance(v, list):
        abort(400, description=f"{key!r} must be a list, got {type(v).__name__}")
    for i, item in enumerate(v):
        if not isinstance(item, str):
            abort(400, description=f"{key!r}[{i}] must be a string, got {type(item).__name__}")
    if len(v) < min_items:
        abort(400, description=f"{key!r} must have at least {min_items} entries, got {len(v)}")
    return v


def _field_number(body: dict, key: str, *, default: Optional[float] = None,
                  min_value: Optional[float] = None,
                  max_value: Optional[float] = None,
                  as_int: bool = False) -> Optional[float]:
    v = body.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        abort(400, description=f"{key!r} must be a number, got {type(v).__name__}")
    if min_value is not None and v < min_value:
        abort(400, description=f"{key!r} must be >= {min_value}")
    if max_value is not None and v > max_value:
        abort(400, description=f"{key!r} must be <= {max_value}")
    return int(v) if as_int else float(v)


def _field_bool_or_none(body: dict, key: str) -> Optional[bool]:
    v = body.get(key)
    if v is None:
        return None
    if not isinstance(v, bool):
        abort(400, description=f"{key!r} must be a boolean, got {type(v).__name__}")
    return v


def _field_enum(body: dict, key: str, allowed: Tuple[str, ...], *,
                default: Optional[str] = None) -> Optional[str]:
    v = body.get(key, default)
    if v is None:
        return default
    if not isinstance(v, str) or v not in allowed:
        abort(400, description=f"{key!r} must be one of {list(allowed)}, got {v!r}")
    return v


def _contained_in(path: Path, root: Path) -> bool:
    """True iff `path` is `root` or a descendant of it.

    Replaces the earlier `str(path).startswith(str(root))` style check, which
    accepted sibling directories whose name happened to share the same prefix
    (e.g. root=/opt/mergese/checkpoints, path=/opt/mergese/checkpoints_private
    matched the prefix). `Path.relative_to` does the real component-wise
    containment check and treats resolved symlink escapes correctly too.
    """
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_model_ref(ref: str, owner: Optional[str] = None) -> str:
    """Turn a frontend model reference into a path/HF id the CLI can consume.

    Accepted forms (all resolved at request time):
        upload://<token>      -> /app/uploads/<token>
        server://<name>       -> <MERGESE_CHECKPOINTS>/<name>
        job://<id>[/<sub>]    -> <ARTIFACTS>/<id>/<sub>  (sub: merged|exported)
        hf://<org>/<model>    -> <org>/<model>
        microsoft/codebert    -> HF id (default for bare strings)
        /abs/path             -> absolute path (ONLY when ALLOW_LOCAL_PATHS=1)

    Any path returned is verified to exist and be contained within its
    expected root - uploads can't escape UPLOADS_ROOT, server refs can't
    escape CHECKPOINTS_ROOT, etc.

    `owner` is the authenticated caller's client id (None when auth is off).
    `upload://` and `job://` refs are refused for anyone other than the owner,
    with the same error message as a missing artifact so existence does not
    leak between callers.
    """
    if not ref:
        raise ValueError("empty model reference")

    if ref.startswith("upload://"):
        token = ref[len("upload://"):].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", token):
            raise ValueError(f"bad upload token: {token!r}")
        root = UPLOADS_ROOT.resolve()
        path = (root / token).resolve()
        if not _contained_in(path, root):
            raise ValueError("upload path escapes uploads root")
        if not path.exists() or not _owns(path, owner):
            raise ValueError(f"upload not found: {token}")
        return str(path)

    if ref.startswith("server://"):
        name = ref[len("server://"):].strip()
        if not CHECKPOINTS_ROOT:
            raise ValueError("server-side checkpoints are not configured")
        root = Path(CHECKPOINTS_ROOT).resolve()
        path = (root / name).resolve()
        if not _contained_in(path, root):
            raise ValueError("server path escapes checkpoints root")
        if not path.exists():
            raise ValueError(f"server checkpoint not found: {name}")
        return str(path)

    if ref.startswith("job://"):
        # job://<id>            -> artifacts/<id>/merged or /exported (auto)
        # job://<id>/merged     -> explicit
        # job://<id>/exported   -> explicit
        rest = ref[len("job://"):].strip().strip("/")
        parts = rest.split("/", 1)
        jid = parts[0]
        if not re.fullmatch(r"[A-Za-z0-9]{6,32}", jid):
            raise ValueError(f"bad job id in ref: {jid!r}")
        sub = parts[1] if len(parts) > 1 else None
        art_root = ARTIFACTS_ROOT.resolve()
        root = art_root / jid
        if not root.exists() or not _owns(root, owner):
            raise ValueError(f"job artifact not found: {jid}")
        if sub:
            if sub not in ("merged", "exported"):
                raise ValueError(f"job ref sub-path must be 'merged' or 'exported': {sub!r}")
            cand = (root / sub).resolve()
        else:
            cand = (root / "merged").resolve()
            if not cand.exists():
                cand = (root / "exported").resolve()
        if not _contained_in(cand, art_root):
            raise ValueError("job ref escapes artifacts root")
        if not cand.exists():
            raise ValueError(f"job output not found for {jid}")
        return str(cand)

    if ref.startswith("hf://"):
        # Everything after the prefix must look like an HF Hub id (org/name).
        # Without this check, `hf:///tmp/evil` would be returned verbatim and
        # fed into AutoModel.from_pretrained as an absolute path - defeating
        # ALLOW_LOCAL_PATHS=0.
        hf_id = ref[len("hf://"):]
        if not _HF_ID.match(hf_id) or "/" not in hf_id or hf_id.startswith("/"):
            raise ValueError(f"not a HuggingFace Hub id: {hf_id!r}")
        return hf_id

    # Absolute paths: only when the operator explicitly opts in
    if ref.startswith("/") or ref.startswith("\\") or ":" in ref[:3]:
        if not ALLOW_LOCAL_PATHS:
            raise ValueError(
                "local filesystem paths are disabled on this server. "
                "Use upload:// or a HuggingFace Hub ID instead, or set "
                "MERGESE_ALLOW_LOCAL_PATHS=1 if you trust your visitors."
            )
        return ref

    # Bare strings: treat as HF Hub IDs (the CLI also tolerates this)
    if not _HF_ID.match(ref):
        raise ValueError(f"unrecognised model reference: {ref!r}")
    return ref


# ---- dataset-reference resolution -------------------------------------------

# Accepted dataset reference forms:
#   bundled://<name>         -> data/benchmarks/<file from index.json>
#   dataset://<token>        -> uploads/_datasets/<token>/data.csv
#   server-dataset://<name>  -> MERGESE_DATASETS/<name>(.csv)
#   hf-dataset://<id>[#split=test][#columns=code1,code2,label]
#                            -> datasets.load_dataset(id) -> mapped to CSV at request time
#   /abs/path/to.csv         -> only when ALLOW_LOCAL_PATHS=1

def resolve_dataset_ref(ref: str, owner: Optional[str] = None) -> str:
    """Resolve a frontend dataset reference into a CSV path the CLI can read.

    `owner` is the authenticated caller's client id (None when auth is off).
    `dataset://` refs are refused for anyone other than the uploader, with the
    same error as a missing dataset so existence does not leak between callers.
    """
    if not ref:
        raise ValueError("empty dataset reference")

    if ref.startswith("bundled://"):
        name = ref[len("bundled://"):].strip()
        idx = _load_benchmarks_index()
        match = next((b for b in idx.get("benchmarks", []) if b["name"] == name), None)
        if not match:
            raise ValueError(
                f"bundled dataset {name!r} not found. "
                f"Known: {[b['name'] for b in idx.get('benchmarks', [])]}"
            )
        path = (BENCHMARKS_ROOT / match["file"]).resolve()
        if not path.exists():
            raise ValueError(f"bundled file missing on disk: {path}")
        return str(path)

    if ref.startswith("dataset://"):
        token = ref[len("dataset://"):].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", token):
            raise ValueError(f"bad dataset token: {token!r}")
        root = DATASET_UPLOADS_ROOT.resolve()
        dir_path = (root / token).resolve()
        path = (dir_path / "data.csv").resolve()
        if not _contained_in(path, root):
            raise ValueError("dataset path escapes uploads root")
        if not path.exists() or not _owns(dir_path, owner):
            raise ValueError(f"uploaded dataset not found: {token}")
        return str(path)

    if ref.startswith("server-dataset://"):
        name = ref[len("server-dataset://"):].strip()
        if not DATASETS_ROOT:
            raise ValueError("server-side datasets are not configured (MERGESE_DATASETS unset)")
        root = Path(DATASETS_ROOT).resolve()
        # Try direct path, then with .csv
        cand = (root / name)
        if not cand.exists() and not cand.suffix:
            cand = root / f"{name}.csv"
        cand = cand.resolve()
        if not _contained_in(cand, root):
            raise ValueError("server-dataset path escapes datasets root")
        if not cand.exists():
            raise ValueError(f"server-dataset not found: {name}")
        return str(cand)

    if ref.startswith("hf-dataset://"):
        # Materialise the HF dataset to a CSV on demand.
        # Format: hf-dataset://<id>[#split=test][#columns=code1,code2,label]
        return _materialise_hf_dataset(ref[len("hf-dataset://"):])

    # Absolute path: only when explicitly allowed
    if ref.startswith("/") or ref.startswith("\\"):
        if not ALLOW_LOCAL_PATHS:
            raise ValueError("local filesystem dataset paths are disabled on this server")
        return ref

    raise ValueError(f"unrecognised dataset reference: {ref!r}")


_HF_DATASET_CACHE = ROOT / "uploads" / "_hf_datasets"
_HF_DATASET_CACHE.mkdir(parents=True, exist_ok=True)


def _materialise_hf_dataset(spec: str) -> str:
    """Fetch a HuggingFace dataset and dump the chosen split as CSV.

    spec format (everything after `hf-dataset://`):
        <id>[#split=<name>][#columns=a,b,c]
    """
    # parse fragments
    parts = spec.split("#")
    dataset_id = parts[0]
    opts: Dict[str, str] = {}
    for kv in parts[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            opts[k.strip()] = v.strip()
    split = opts.get("split", "test")
    columns = opts.get("columns")  # comma-separated mapping override

    # cache key
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{dataset_id}__{split}__{columns or 'auto'}")
    out_csv = _HF_DATASET_CACHE / f"{key}.csv"
    if out_csv.exists():
        return str(out_csv)

    try:
        from datasets import load_dataset
    except ImportError:
        raise ValueError(
            "Loading HuggingFace datasets requires the `datasets` package. "
            "Install: pip install datasets"
        )

    try:
        ds = load_dataset(dataset_id, split=split)
    except Exception as e:
        raise ValueError(f"could not load HF dataset {dataset_id!r}: {e}")

    # Decide which columns to write. If the user supplied a mapping, use it;
    # otherwise auto-detect (code1+code2+label, or func+target, or code+label).
    cols = [c.strip() for c in columns.split(",")] if columns else None
    if not cols:
        feats = set(ds.column_names)
        if {"code1", "code2", "label"} <= feats:
            cols = ["code1", "code2", "label"]
        elif {"func1", "func2", "label"} <= feats:
            cols = ["func1", "func2", "label"]
        elif {"func", "target"} <= feats:
            cols = ["func", "target"]   # Devign
        elif {"code", "label"} <= feats:
            cols = ["code", "label"]
        else:
            raise ValueError(
                f"can't auto-map columns of HF dataset {dataset_id!r} "
                f"(available: {ds.column_names}). "
                "Specify: hf-dataset://<id>#columns=code1,code2,label"
            )

    pair_mode = len(cols) >= 3
    import csv as _csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if pair_mode:
            w.writerow(["code1", "code2", "label"])
            for r in ds:
                w.writerow([str(r[cols[0]])[:8000], str(r[cols[1]])[:8000], int(r[cols[2]])])
        else:
            w.writerow(["code", "label"])
            for r in ds:
                w.writerow([str(r[cols[0]])[:8000], int(r[cols[1]])])
    return str(out_csv)


# ---- helpers -----------------------------------------------------------------

def _build_cmd(args: List[str]) -> List[str]:
    base = shlex.split(MERGESE_BIN)
    return [*base, *args]


def _allocate_job_id() -> Tuple[str, Path]:
    """Pre-allocate a job id + artifact dir so callers can place output paths
    (e.g. merge output_dir) under the same id that _new_job will use later."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = ARTIFACTS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def _new_job(kind: str, cli_args: List[str], params: dict,
             result_basename: Optional[str] = None,
             job_id: Optional[str] = None,
             owner: Optional[str] = None) -> Job:
    if job_id is None:
        job_id, job_dir = _allocate_job_id()
    else:
        job_dir = ARTIFACTS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
    # Stamp the owner onto the artifact dir so job:// refs can be ownership-
    # checked after a server restart, when the in-memory JOBS dict is empty.
    _write_owner(job_dir, owner)
    log_path = job_dir / "log.txt"
    result_path = (job_dir / result_basename) if result_basename else None

    full_args = list(cli_args)
    if result_basename and result_basename.endswith(".json"):
        full_args.extend(["--json-out", str(result_path)])

    cmd = _build_cmd(full_args)
    job = Job(
        id=job_id,
        cmd=cmd,
        kind=kind,
        params=params,
        log_path=log_path,
        result_path=result_path,
        needs_network=_cmd_needs_network(cmd),
        owner=owner,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def _hf_cache_root() -> Path:
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    legacy = os.environ.get("TRANSFORMERS_CACHE")
    if legacy:
        return Path(legacy)
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_id_is_cached(hf_id: str) -> bool:
    """True if a HuggingFace Hub id already has a local snapshot.

    Uses the standard `models--org--name` cache layout so we can tell an
    offline-runnable job (cached) from one that would need a Hub download.
    """
    folder = "models--" + hf_id.replace("/", "--")
    snap = _hf_cache_root() / folder / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


def _cmd_needs_network(cmd: List[str]) -> bool:
    """Does this resolved command reference an uncached Hub id?

    Resolved local refs are absolute paths; a bare `org/model` token that is
    neither an existing path nor already cached implies a Hub download, which
    the offline worker cannot perform.
    """
    for tok in cmd:
        if tok.startswith("-") or os.path.isabs(tok) or os.sep in tok and Path(tok).exists():
            continue
        if _HF_ID.match(tok) and "/" in tok and not Path(tok).exists():
            if not _hf_id_is_cached(tok):
                return True
    return False


def _uncached_hf_ids_in_cmd(cmd: List[str]) -> List[str]:
    """Return the uncached HF ids present in `cmd`, preserving order & dedup."""
    out, seen = [], set()
    for tok in cmd:
        if tok in seen:
            continue
        if tok.startswith("-") or os.path.isabs(tok) or os.sep in tok and Path(tok).exists():
            continue
        if _HF_ID.match(tok) and "/" in tok and not Path(tok).exists():
            if not _hf_id_is_cached(tok):
                out.append(tok); seen.add(tok)
    return out


def _cached_hf_snapshot_path(hf_id: str) -> Optional[str]:
    """Resolve a cached HF id to the local snapshot path (`refs/main` revision
    when present, else any snapshot). Returns None if the id isn't cached."""
    folder = "models--" + hf_id.replace("/", "--")
    model_dir = _hf_cache_root() / folder
    ref = model_dir / "refs" / "main"
    try:
        if ref.exists():
            rev = ref.read_text().strip()
            snap = model_dir / "snapshots" / rev
            if snap.is_dir():
                return str(snap)
    except OSError:
        pass
    snaps = model_dir / "snapshots"
    if snaps.is_dir():
        for s in sorted(snaps.iterdir()):
            if s.is_dir():
                return str(s)
    return None


def _rewrite_hf_ids_to_local_paths(cmd: List[str]) -> List[str]:
    """Replace every cached HF id in `cmd` with its local snapshot path.

    `AutoModel.from_pretrained(<path>)` works under TRANSFORMERS_OFFLINE on
    every transformers version we care about, while `from_pretrained(<id>)`
    fails to resolve the cache in some v4.x releases. Rewriting the cmd is
    a cheap belt-and-suspenders against that regression. Uncached ids are
    left as-is so the "needs network" refusal path can still catch them.
    """
    out = []
    for tok in cmd:
        if (
            tok.startswith("-")
            or os.path.isabs(tok)
            or (os.sep in tok and Path(tok).exists())
        ):
            out.append(tok); continue
        if _HF_ID.match(tok) and "/" in tok and not Path(tok).exists():
            local = _cached_hf_snapshot_path(tok)
            if local:
                out.append(local); continue
        out.append(tok)
    return out


# ---- HF auto-prefetch --------------------------------------------------------
#
# The merge worker runs inside a stripped network namespace, so by itself it
# cannot download a model from the Hub. But the Flask process DOES have network,
# so when a job references an HF id that isn't cached yet we can grab it in the
# parent before the sandboxed child starts. This lets a visitor type any public
# HF Hub id (not just the pre-cached suggestions) and have it work.
#
# Guards:
#   - Only `org/name` shaped tokens are considered (`_HF_ID.match` + "/" in it).
#   - A per-model size cap stops a visitor from hooking us into a 50 GB llama
#     checkpoint. Set MERGESE_HF_PREFETCH_MAX_BYTES=0 to disable the cap.
#   - MERGESE_HF_AUTO_PREFETCH=0 reverts to the old "refuse offline" behaviour.
#   - The job stays in `running` state throughout the download; progress goes to
#     the job log file so visitors can watch it stream.
HF_AUTO_PREFETCH = bool(int(os.environ.get("MERGESE_HF_AUTO_PREFETCH", "1")))
HF_PREFETCH_MAX_BYTES = int(
    os.environ.get("MERGESE_HF_PREFETCH_MAX_BYTES", str(5 * 1024 ** 3))  # 5 GB
)


def _prefetch_hf_id(hf_id: str, log) -> str:
    """Download `hf_id` into HF_HOME synchronously, return the local snapshot
    path.

    The returned path is handed back so the caller can rewrite the worker's
    cmd to load from a filesystem path rather than the HF id. We do this
    because some combinations of transformers + huggingface_hub refuse to
    resolve an id -> local snapshot under TRANSFORMERS_OFFLINE=1 (even when
    the files are plainly in HF_HOME/hub); the sandboxed worker has no
    network to retry, and `AutoModel.from_pretrained(path)` always works.

    Writes progress lines into the open `log` file so a visitor can follow it
    through /api/jobs/<id>/stream. Raises `ValueError` with an actionable
    message on any failure (lookup failed, too large, download failed) so the
    caller can mark the job as `error` with a useful summary.
    """
    from huggingface_hub import HfApi, snapshot_download
    log.write(f"[prefetch] looking up {hf_id} on HuggingFace Hub...\n".encode())
    try:
        info = HfApi().model_info(hf_id, files_metadata=True)
    except Exception as e:
        raise ValueError(f"could not look up {hf_id} on HF Hub: {e}")
    sizes = [getattr(f, "size", None) or 0 for f in (getattr(info, "siblings", []) or [])]
    total = sum(sizes)
    gb = total / (1024 ** 3)
    if HF_PREFETCH_MAX_BYTES > 0 and total > HF_PREFETCH_MAX_BYTES:
        cap_gb = HF_PREFETCH_MAX_BYTES / (1024 ** 3)
        raise ValueError(
            f"refusing to auto-prefetch {hf_id}: "
            f"{gb:.1f} GB > {cap_gb:.1f} GB operator limit. "
            f"Ask the operator to pre-cache this model, or upload safetensors directly."
        )
    log.write(
        f"[prefetch] downloading {hf_id} ({gb:.2f} GB, {len(sizes)} files)...\n"
        .encode()
    )
    try:
        local_path = snapshot_download(hf_id)
    except Exception as e:
        raise ValueError(f"could not download {hf_id}: {e}")
    log.write(f"[prefetch] done: {hf_id} -> {local_path}\n".encode())
    return local_path


def _rlimit_preexec():
    """Applied in the worker child before exec: rlimits + new session.

    A new session (setsid) puts the worker in its own process group so a
    timeout can kill the whole tree, including any threads/children it spawned.
    """
    os.setsid()
    def _set(res, soft):
        try:
            hard = resource.getrlimit(res)[1]
            cap = soft if hard == resource.RLIM_INFINITY else min(soft, hard)
            resource.setrlimit(res, (cap, hard))
        except (ValueError, OSError):
            pass
    if WORKER_CPU_SEC > 0:
        _set(resource.RLIMIT_CPU, WORKER_CPU_SEC)
    if WORKER_MAX_FILE_BYTES > 0:
        _set(resource.RLIMIT_FSIZE, WORKER_MAX_FILE_BYTES)
    if WORKER_MAX_NPROC > 0:
        _set(resource.RLIMIT_NPROC, WORKER_MAX_NPROC)
    if WORKER_AS_BYTES > 0:
        _set(resource.RLIMIT_AS, WORKER_AS_BYTES)


def _build_worker(job: Job, logf) -> subprocess.Popen:
    """Launch the CLI for `job` in a hardened child process."""
    job_dir = ARTIFACTS_ROOT / job.id
    home = job_dir / "home"
    tmp = job_dir / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # Stripped environment: only what the merge CLI actually needs. The full
    # server environment (secrets, DB URLs, tokens) is never inherited.
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "FORCE_COLOR": "0",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    # Read-only access to the shared model cache so already-fetched encoders
    # resolve offline. Passing the path grants read of the cache, not network.
    for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    if WORKER_OFFLINE:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    cmd = list(job.cmd)
    # No network unless the operator has opted out of offline mode. The
    # namespace also blocks localhost, so the worker cannot reach the Flask
    # service, a database, or cloud metadata endpoints.
    if WORKER_OFFLINE and NETNS_OK:
        cmd = [_UNSHARE_BIN, "-rn", "--", *cmd]

    return subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=str(job_dir),          # job-private working dir, never the app source
        env=env,
        preexec_fn=_rlimit_preexec,
        close_fds=True,
    )


def _run_job(job: Job) -> None:
    with RUN_SEMA:
        # Honour a cancel issued while the job was queued: the user cancelled
        # before our turn in the RUN_SEMA, so we just don't launch the worker.
        with JOBS_LOCK:
            if job.status == "cancelled":
                job.finished_at = job.finished_at or time.time()
                return
            job.status = "running"
            job.started_at = time.time()
        try:
            with open(job.log_path, "wb", buffering=0) as logf:
                # If the job wants an uncached HF id and auto-prefetch is on,
                # grab it in the Flask parent (which has network) before the
                # sandboxed worker starts. On failure, write the reason to the
                # log and mark the job errored - DO NOT fall through to the
                # worker (which would just hit an opaque offline-cache miss).
                if job.needs_network and WORKER_OFFLINE and NETNS_OK:
                    if HF_AUTO_PREFETCH:
                        try:
                            for hf_id in _uncached_hf_ids_in_cmd(job.cmd):
                                _prefetch_hf_id(hf_id, logf)
                            # Re-check: everything we asked for should now be
                            # cached.
                            job.needs_network = _cmd_needs_network(job.cmd)
                        except ValueError as e:
                            logf.write(f"[prefetch] FAILED: {e}\n".encode())
                            with JOBS_LOCK:
                                job.status = "error"
                                job.error = str(e)
                                job.finished_at = time.time()
                            return
                    if job.needs_network:
                        msg = ("this job references a HuggingFace model that is not "
                               "cached on the server. The merge worker runs offline, "
                               "so ask the operator to pre-fetch the model, or upload "
                               "safetensors weights directly.")
                        logf.write(f"[mergese] refused: {msg}\n".encode())
                        with JOBS_LOCK:
                            job.status = "error"
                            job.error = msg
                            job.finished_at = time.time()
                        return
                # Rewrite cached HF ids -> local snapshot paths. Always safe:
                # `AutoModel.from_pretrained(<path>)` works offline regardless
                # of transformers version, while `from_pretrained(<id>)` under
                # TRANSFORMERS_OFFLINE fails on some v4.x releases.
                job.cmd = _rewrite_hf_ids_to_local_paths(job.cmd)
                proc = _build_worker(job, logf)
                with JOBS_LOCK:
                    job.pid = proc.pid
                    JOB_PROCS[job.id] = proc
                try:
                    rc = proc.wait(timeout=WORKER_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    _kill_proc_group(proc)
                    rc = proc.wait()
                    with JOBS_LOCK:
                        job.status = "error"
                        job.error = f"job exceeded the {WORKER_TIMEOUT_SEC}s time limit"
                        job.finished_at = time.time()
                    logf.write(f"\n[mergese] killed: exceeded {WORKER_TIMEOUT_SEC}s limit\n"
                               .encode())
                    return
                with JOBS_LOCK:
                    job.exit_code = rc
                    job.finished_at = time.time()
                    if job.status == "cancelled":
                        pass
                    elif rc == 0:
                        job.status = "done"
                    else:
                        job.status = "error"
                        job.error = f"process exited with code {rc}"
        except Exception as e:
            with JOBS_LOCK:
                job.status = "error"
                job.error = str(e)
                job.finished_at = time.time()
        finally:
            with JOBS_LOCK:
                JOB_PROCS.pop(job.id, None)


def _kill_proc_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the worker's whole process group."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def _tail_stream(job: Job):
    """Generator yielding SSE messages by tailing the job's log file."""
    # Wait for log file to exist briefly
    for _ in range(50):
        if job.log_path.exists():
            break
        time.sleep(0.05)
    if not job.log_path.exists():
        yield "event: error\ndata: log not available\n\n"
        return

    with open(job.log_path, "rb") as f:
        while True:
            line = f.readline()
            if line:
                payload = line.decode("utf-8", errors="replace").rstrip("\n")
                yield f"data: {json.dumps({'line': payload})}\n\n"
                continue
            # End of file - check job status
            with JOBS_LOCK:
                status = job.status
            if status in ("done", "error", "cancelled"):
                yield f"event: end\ndata: {json.dumps({'status': status, 'exit_code': job.exit_code})}\n\n"
                return
            time.sleep(0.2)


# ---- Flask app ---------------------------------------------------------------

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
# Don't 405/redirect on a trailing slash - visitors curl with and without,
# and Flask's default sends POST/PUT/DELETE to a 405 for the slash variant.
app.url_map.strict_slashes = False
CORS(app)


# ---- security response headers ----------------------------------------------
#
# The deployment runs behind nginx + TLS, but setting these at the application
# layer means they apply the same way whether the app is served through nginx,
# gunicorn direct, or `python server/app.py` in dev. The CSP is intentionally
# narrow: the only off-origin resource the UI loads is the Cloudflare
# Turnstile widget, which must be explicitly allow-listed in script-src and
# frame-src for the challenge iframe to render.
_CSP = (
    "default-src 'self'; "
    "img-src 'self' data:; "
    "style-src 'self' 'unsafe-inline'; "
    "script-src 'self' https://challenges.cloudflare.com; "
    "frame-src https://challenges.cloudflare.com; "
    "connect-src 'self' https://challenges.cloudflare.com; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("Content-Security-Policy", _CSP)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    # Only advertise HSTS on an HTTPS request. In dev the app may be hit over
    # plain HTTP; setting HSTS there would still work but is unhelpful noise.
    if request.is_secure or request.headers.get("X-Forwarded-Proto") == "https":
        resp.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )
    return resp


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND), "index.html")


@app.route("/<path:asset>")
def static_assets(asset: str):
    candidate = FRONTEND / asset
    if candidate.exists() and candidate.is_file():
        return send_from_directory(str(FRONTEND), asset)
    abort(404)


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "version": _cli_version()})


def _cli_version() -> str:
    try:
        out = subprocess.check_output(_build_cmd(["--version"]), text=True, timeout=5)
        return out.strip()
    except Exception as e:
        return f"unknown ({e})"


# ---- /api/inspect ------------------------------------------------------------

@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    owner = _authenticate()
    body = _json_body()
    models = _field_str_list(body, "models", min_items=2)
    base = _field_str(body, "base")
    over = _capacity_response()
    if over:
        return over
    try:
        resolved_models = [resolve_model_ref(m, owner=owner) for m in models]
        resolved_base = resolve_model_ref(base, owner=owner) if base else None
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    _reserve_job_quota()
    args = ["inspect", *resolved_models]
    if resolved_base:
        args.extend(["--base", resolved_base])
    job = _new_job("inspect", args, {"models": models, "base": base},
                   result_basename="report.json", owner=owner)
    return jsonify({"job_id": job.id, "status": job.status}), 202


# ---- /api/merge --------------------------------------------------------------

@app.route("/api/merge", methods=["POST"])
def api_merge():
    owner = _authenticate()
    body = _json_body()
    models = _field_str_list(body, "models", min_items=2)
    base = _field_str(body, "base", required=True)
    method = _field_enum(body, "method",
                         ("ties", "dare-ties", "wudi", "pcb", "average"),
                         default="ties")
    trim_percentile = _field_number(body, "trim_percentile", default=20.0,
                                    min_value=0.0, max_value=99.0)
    drop_rate = _field_number(body, "drop_rate", default=0.3,
                              min_value=0.0, max_value=0.99)
    wudi_steps = _field_number(body, "wudi_steps", min_value=1, max_value=100000, as_int=True)
    wudi_lr = _field_number(body, "wudi_lr", min_value=0.0, max_value=1.0)
    pcb_ratio = _field_number(body, "pcb_ratio", min_value=0.01, max_value=1.0)
    pcb_lambda = _field_number(body, "pcb_lambda", min_value=0.0, max_value=1000.0)
    pcb_scope = _field_enum(body, "pcb_scope", ("global", "tensor"))
    weights = _field_str(body, "weights")
    seed = _field_number(body, "seed", default=42, min_value=0, max_value=2**31 - 1, as_int=True)
    task = _field_str(body, "task", default="") or ""
    encoder_only = _field_bool_or_none(body, "encoder_only")
    over = _capacity_response()
    if over:
        return over

    try:
        resolved_models = [resolve_model_ref(m, owner=owner) for m in models]
        resolved_base = resolve_model_ref(base, owner=owner)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    _reserve_job_quota()
    job_id, job_dir = _allocate_job_id()
    _write_owner(job_dir, owner)
    out_dir = job_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = ["merge", *resolved_models,
            "--base", resolved_base,
            "--method", method,
            "--trim-percentile", str(trim_percentile),
            "--drop-rate", str(drop_rate),
            "--seed", str(seed),
            "--output", str(out_dir)]
    if wudi_steps is not None:
        args.extend(["--wudi-steps", str(wudi_steps)])
    if wudi_lr is not None:
        args.extend(["--wudi-lr", str(wudi_lr)])
    if pcb_ratio is not None:
        args.extend(["--pcb-ratio", str(pcb_ratio)])
    if pcb_lambda is not None:
        args.extend(["--pcb-lambda", str(pcb_lambda)])
    if pcb_scope:
        args.extend(["--pcb-scope", str(pcb_scope)])
    if weights:
        args.extend(["--weights", weights])
    if task:
        args.extend(["--task", task])
    if encoder_only is True:
        args.append("--encoder-only")
    elif encoder_only is False:
        args.append("--include-heads")

    job = _new_job("merge", args, body, job_id=job_id, owner=owner)
    job.artifacts["output_dir"] = str(out_dir)
    return jsonify({"job_id": job.id, "status": job.status, "output_dir": str(out_dir)}), 202


# ---- /api/evaluate -----------------------------------------------------------

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    owner = _authenticate()
    body = _json_body()
    model = _field_str(body, "model", required=True)
    task = _field_str(body, "task", default="clone_detection")
    dataset = _field_str(body, "dataset")
    test_file = _field_str(body, "test_file")
    batch_size = _field_number(body, "batch_size", default=32, min_value=1, max_value=4096, as_int=True)
    max_length = _field_number(body, "max_length", default=512, min_value=1, max_value=16384, as_int=True)
    limit = _field_number(body, "limit", default=0, min_value=0, max_value=10 ** 9, as_int=True)
    metric = _field_enum(body, "metric", ("auto", "binary", "macro"), default="auto")
    over = _capacity_response()
    if over:
        return over

    try:
        resolved_model = resolve_model_ref(model, owner=owner)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Unified dataset reference handling: the frontend sends one `dataset_ref`
    # using bundled:// / dataset:// / hf-dataset:// / server-dataset://. Older
    # clients can still send `dataset` + `test_file` as raw strings.
    args = ["evaluate", resolved_model, "--task", task,
            "--batch-size", str(batch_size),
            "--max-length", str(max_length),
            "--limit", str(limit),
            "--metric", metric]

    dataset_ref = body.get("dataset_ref")
    if dataset_ref:
        try:
            csv_path = resolve_dataset_ref(dataset_ref, owner=owner)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        args.extend(["--test-file", csv_path])
    else:
        if dataset:
            args.extend(["--dataset", dataset])
        if test_file:
            args.extend(["--test-file", test_file])

    _reserve_job_quota()
    job = _new_job("evaluate", args, body, result_basename="metrics.json", owner=owner)
    return jsonify({"job_id": job.id, "status": job.status}), 202


# ---- /api/export -------------------------------------------------------------

@app.route("/api/export", methods=["POST"])
def api_export():
    owner = _authenticate()
    body = _json_body()
    model = _field_str(body, "model", required=True)
    fmt = _field_enum(body, "format",
                      ("huggingface", "onnx", "torchscript"),
                      default="huggingface")
    over = _capacity_response()
    if over:
        return over

    try:
        resolved_model = resolve_model_ref(model, owner=owner)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    _reserve_job_quota()
    job_id, base_out = _allocate_job_id()
    _write_owner(base_out, owner)
    if fmt == "huggingface":
        out_path = base_out / "exported"
    elif fmt == "onnx":
        # torch.onnx writes external-data files (model.onnx + sibling blobs)
        # once the model exceeds the 2 GB protobuf limit. Writing to a
        # dedicated subdir keeps those siblings together so the download zip
        # can include every piece - a bare `model.onnx` file would lose the
        # external data and fail onnx.checker on extraction.
        onnx_dir = base_out / "exported"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        out_path = onnx_dir / "model.onnx"
    else:
        ts_dir = base_out / "exported"
        ts_dir.mkdir(parents=True, exist_ok=True)
        out_path = ts_dir / "model.pt"

    args = ["export", resolved_model, "--format", fmt, "--output", str(out_path)]
    job = _new_job("export", args, body, job_id=job_id, owner=owner)
    job.artifacts["output"] = str(out_path)
    return jsonify({"job_id": job.id, "status": job.status, "output": str(out_path)}), 202


# ---- /api/jobs ---------------------------------------------------------------

@app.route("/api/jobs")
def api_jobs():
    # When auth is on, callers see only their own jobs.
    owner = None
    if REQUIRE_AUTH:
        client = _resolve_client()
        owner = client.client_id if client else "\x00none"
    with JOBS_LOCK:
        items = [j.to_dict() for j in JOBS.values()
                 if owner is None or j.owner == owner]
    items.sort(key=lambda x: x.get("started_at") or 0, reverse=True)
    return jsonify({"jobs": items})


@app.route("/api/jobs/<job_id>")
def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404)
    _require_owner(job)
    return jsonify(job.to_dict())


@app.route("/api/jobs/<job_id>/stream")
def api_job_stream(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404)
    _require_owner(job)
    return Response(_tail_stream(job), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/jobs/<job_id>/result")
def api_job_result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404)
    _require_owner(job)
    if not job.result_path or not job.result_path.exists():
        return jsonify({"status": job.status, "result": None}), 200
    try:
        data = json.loads(job.result_path.read_text())
    except Exception as e:
        return jsonify({"error": f"could not parse result: {e}"}), 500
    return jsonify({"status": job.status, "result": data})


@app.route("/api/jobs/<job_id>/log")
def api_job_log(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404)
    _require_owner(job)
    if not job.log_path.exists():
        return Response("", mimetype="text/plain")
    return Response(job.log_path.read_text(errors="replace"), mimetype="text/plain")


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_job_cancel(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        proc = JOB_PROCS.get(job_id)
    if not job:
        abort(404)
    _require_owner(job)
    if proc and proc.poll() is None:
        try:
            # The worker runs in its own session (see _rlimit_preexec), so we
            # signal the whole process group - killing just the unshare parent
            # would orphan the python child doing the actual merge.
            with JOBS_LOCK:
                job.status = "cancelled"
            _kill_proc_group(proc)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        with JOBS_LOCK:
            job.finished_at = time.time()
        return jsonify({"ok": True, "status": "cancelled"})
    # Pending jobs: no process has started yet, but _run_job will check this
    # status flag before launching the subprocess. Marking the job cancelled
    # here means the worker simply never starts.
    if job.status == "pending":
        with JOBS_LOCK:
            job.status = "cancelled"
            job.finished_at = time.time()
        try:
            job.log_path.write_text("[mergese] cancelled by user while queued\n")
        except OSError:
            pass
        return jsonify({"ok": True, "status": "cancelled"})
    return jsonify({"ok": False, "status": job.status})


# ---- /api/checkpoints --------------------------------------------------------

@app.route("/api/checkpoints")
def api_checkpoints():
    """List checkpoint directories under MERGESE_CHECKPOINTS (if configured).

    These are operator-mounted, globally visible to anyone who can run a merge,
    so the listing is behind the same authentication wall as the compute
    endpoints to avoid advertising server-side filesystem layout anonymously.
    """
    _authenticate()
    if not CHECKPOINTS_ROOT:
        return jsonify({"root": None, "entries": [], "note": "MERGESE_CHECKPOINTS not set"})
    root = Path(CHECKPOINTS_ROOT)
    if not root.exists():
        return jsonify({"root": str(root), "entries": [], "note": "root does not exist"})
    entries = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            entries.append({
                "name": p.name,
                "path": str(p),
                "files": [f.name for f in p.iterdir()][:32],
            })
    return jsonify({"root": str(root), "entries": entries})


# ---- /api/presets ------------------------------------------------------------

PRESETS_FILE = HERE / "presets.json"

@app.route("/api/tasks")
def api_tasks():
    """Mirror of `mergese tasks --json` so the frontend can populate dropdowns."""
    try:
        out = subprocess.check_output(_build_cmd(["tasks", "--json"]), text=True, timeout=15)
        return Response(out, mimetype="application/json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/presets")
def api_presets():
    """User-editable example workflows (loaded once at request time)."""
    if PRESETS_FILE.exists():
        try:
            return jsonify(json.loads(PRESETS_FILE.read_text()))
        except Exception as e:
            return jsonify({"presets": [], "error": str(e)})
    return jsonify({"presets": []})


# ---- /api/uploads ------------------------------------------------------------

# Files we accept inside an HF checkpoint directory. The presence of config.json
# is required; the rest are best-effort copied through.
_HF_FILES_REQUIRED = {"config.json"}
_HF_FILES_OPTIONAL = {
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "model.safetensors", "pytorch_model.bin", "preprocessor_config.json",
}
_HF_FILES_OPTIONAL_GLOB = ("model-*.safetensors", "pytorch_model-*.bin")


def _new_upload_token() -> str:
    """A URL-safe token used as both the directory name and the API id."""
    return uuid.uuid4().hex[:16]


def _upload_dir(token: str) -> Path:
    return UPLOADS_ROOT / token


def _scan_unsafe_files(p: Path) -> Optional[str]:
    """Reject uploads carrying pickle checkpoints or executable code.

    The merge worker deserializes whatever weights it is handed. Pickle-backed
    formats (`.bin`, `.pt`, `.ckpt`, ...) run arbitrary code on load, and code
    files (`.py`, `.so`, ...) have no place in a tensor-only checkpoint. Unless
    an operator has explicitly opted into pickle uploads, both are refused here,
    before anything ever loads them. Returns an error string, or None if clean.
    """
    if ALLOW_PICKLE_UPLOADS:
        return None
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in _PICKLE_EXTS:
            return (f"refusing {f.name}: pickle-based checkpoints can execute code "
                    f"on load. Upload safetensors weights (model.safetensors) instead.")
        if ext in _CODE_EXTS:
            return (f"refusing {f.name}: executable/code files are not allowed in a "
                    f"checkpoint upload.")
    return None


def _validate_hf_dir(p: Path) -> Optional[str]:
    """Return None if `p` looks like a safe HF checkpoint dir, else an error."""
    if not (p / "config.json").exists():
        return f"missing config.json in {p.name}"
    # Safe-format gate first: an uploaded pickle checkpoint must never reach the
    # loader, even if it also ships valid safetensors alongside.
    unsafe = _scan_unsafe_files(p)
    if unsafe:
        return unsafe
    has_weights = (p / "model.safetensors").exists() or any(p.glob("model-*.safetensors"))
    if not has_weights and ALLOW_PICKLE_UPLOADS:
        has_weights = (p / "pytorch_model.bin").exists() or any(p.glob("pytorch_model-*.bin"))
    if not has_weights:
        return (f"no safetensors weights found in {p.name} (expected "
                f"model.safetensors). Convert pickle .bin checkpoints to "
                f"safetensors before uploading.")
    return None


def _safe_zip_extract(zf: zipfile.ZipFile, dest: Path) -> Tuple[Optional[str], List[str]]:
    """Extract a zip to `dest`, refusing Zip Slip and absolute-path entries.

    Returns (error_or_None, list_of_extracted_relative_paths).
    """
    dest = dest.resolve()
    extracted: List[str] = []
    members = [m for m in zf.infolist() if not m.filename.endswith("/")]
    # Zip-bomb guard 1: entry count.
    if len(members) > MAX_ARCHIVE_ENTRIES:
        return (f"archive has too many entries ({len(members)} > "
                f"{MAX_ARCHIVE_ENTRIES})"), extracted
    # Zip-bomb guard 2: declared uncompressed total. This reads the central
    # directory only, so it rejects a bomb before a single byte is written.
    total_uncompressed = sum(max(m.file_size, 0) for m in members)
    if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
        return (f"archive expands to {total_uncompressed} bytes, over the "
                f"{MAX_UNCOMPRESSED_BYTES}-byte limit"), extracted
    written = 0
    for member in members:
        name = member.filename
        # Zip-bomb guard 3: per-entry compression ratio (catches a small entry
        # whose header lies about file_size, or a highly compressible payload).
        if member.compress_size > 0:
            ratio = member.file_size / member.compress_size
            if ratio > MAX_COMPRESSION_RATIO:
                return (f"zip entry {name!r} has a suspicious compression ratio "
                        f"({ratio:.0f}:1)"), extracted
        # Strip Windows drive letters and leading slashes
        clean = name.replace("\\", "/").lstrip("/")
        if ".." in clean.split("/"):
            return f"zip entry tries to escape extraction directory: {name!r}", extracted
        target = (dest / clean).resolve()
        if not str(target).startswith(str(dest)):
            return f"zip entry escapes extraction directory: {name!r}", extracted
        target.parent.mkdir(parents=True, exist_ok=True)
        # Zip-bomb guard 4: enforce the running total during the copy, so a
        # lying header cannot slip a bomb past the pre-check above.
        with zf.open(member, "r") as src, open(target, "wb") as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UNCOMPRESSED_BYTES:
                    out.close()
                    return (f"archive exceeded the {MAX_UNCOMPRESSED_BYTES}-byte "
                            f"uncompressed limit during extraction"), extracted
                out.write(chunk)
        extracted.append(clean)
    return None, extracted


def _flatten_single_subdir(p: Path) -> None:
    """If `p` contains exactly one subdir and nothing else, move its contents up.

    Most zipped HF checkpoints contain a single top-level folder like
    `codebert_bcb/` that wraps the files - we hoist them so the upload dir
    itself is a valid HF dir.
    """
    entries = [c for c in p.iterdir() if c.name not in (".DS_Store",)]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in inner.iterdir():
            shutil.move(str(child), str(p / child.name))
        inner.rmdir()


@app.route("/api/uploads", methods=["POST"])
def api_upload():
    """Accept a checkpoint upload.

    Two modes:
      1. Single file with field name `file` - must be a .zip; we extract it.
      2. Multiple files with field name `files` - copied verbatim into one dir.
         At minimum the upload must include config.json and a weights file.

    Returns: { token, ref: "upload://<token>", size, files: [...] }
    """
    owner = _authenticate()
    label = secure_filename((request.form.get("label") or "").strip())[:64] or None

    if "file" in request.files:
        f = request.files["file"]
        name = secure_filename(f.filename or "upload.zip")
        if not name.lower().endswith(".zip"):
            return jsonify({"error": "single-file uploads must be a .zip"}), 400

        token = _new_upload_token()
        d = _upload_dir(token)
        d.mkdir(parents=True, exist_ok=True)
        try:
            zip_path = d / "_upload.zip"
            f.save(str(zip_path))
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    err, extracted = _safe_zip_extract(zf, d)
                    if err:
                        shutil.rmtree(d, ignore_errors=True)
                        return jsonify({"error": err}), 400
            finally:
                zip_path.unlink(missing_ok=True)

            _flatten_single_subdir(d)
            err = _validate_hf_dir(d)
            if err:
                shutil.rmtree(d, ignore_errors=True)
                return jsonify({
                    "error": err,
                    "hint": "ZIP a folder that contains config.json and "
                            "model.safetensors (plus the tokenizer files). "
                            "Pickle .bin checkpoints are refused by default; "
                            "convert them with safetensors first.",
                }), 400

            size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
            if label:
                (d / ".label").write_text(label)
            _write_owner(d, owner)
            return jsonify({
                "token": token,
                "ref": f"upload://{token}",
                "size": size,
                "label": label,
                "files": sorted(p.relative_to(d).as_posix()
                                for p in d.rglob("*") if p.is_file()),
            })
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            return jsonify({"error": f"upload failed: {e}"}), 500

    if "files" in request.files or request.files:
        files = request.files.getlist("files") or list(request.files.values())
        if not files:
            return jsonify({"error": "no files in request"}), 400

        token = _new_upload_token()
        d = _upload_dir(token)
        d.mkdir(parents=True, exist_ok=True)
        try:
            for f in files:
                name = secure_filename(f.filename or "")
                if not name:
                    continue
                ext = Path(name).suffix.lower()
                if not ALLOW_PICKLE_UPLOADS and (ext in _PICKLE_EXTS or ext in _CODE_EXTS):
                    shutil.rmtree(d, ignore_errors=True)
                    return jsonify({
                        "error": f"refusing {name}: pickle checkpoints and code files "
                                 f"are not allowed. Upload model.safetensors instead.",
                    }), 400
                f.save(str(d / name))
            err = _validate_hf_dir(d)
            if err:
                shutil.rmtree(d, ignore_errors=True)
                return jsonify({"error": err}), 400
            size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
            if label:
                (d / ".label").write_text(label)
            _write_owner(d, owner)
            return jsonify({
                "token": token,
                "ref": f"upload://{token}",
                "size": size,
                "label": label,
                "files": sorted(p.name for p in d.iterdir() if p.is_file()),
            })
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            return jsonify({"error": f"upload failed: {e}"}), 500

    return jsonify({"error": "POST a .zip as 'file' or HF files as 'files'"}), 400


@app.route("/api/library")
def api_library():
    """Single-place listing of every model + dataset the user can pick from.

    Models groups:
        uploads, server, jobs, suggestions
    Datasets groups (returned under `datasets`):
        bundled, uploads, server, suggestions

    Uploads, dataset-uploads, and finished-job outputs are scoped to the
    authenticated caller - a visitor only ever sees their own artifacts in
    the picker. Server-mounted checkpoints and bundled datasets are global.
    """
    owner = _authenticate()
    out = {"uploads": [], "server": [], "jobs": [], "suggestions": [],
           "datasets": {"bundled": [], "uploads": [], "server": [], "suggestions": []}}

    # uploads (scoped to caller)
    if UPLOADS_ROOT.exists():
        for d in sorted(UPLOADS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir() or not _owns(d, owner):
                continue
            label_path = d / ".label"
            label = label_path.read_text().strip() if label_path.exists() else None
            size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
            out["uploads"].append({
                "ref": f"upload://{d.name}",
                "label": label or d.name,
                "size": size,
                "mtime": d.stat().st_mtime,
            })

    # server-mounted (global; operator-provided, not user content)
    if CHECKPOINTS_ROOT:
        root = Path(CHECKPOINTS_ROOT)
        if root.exists():
            for p in sorted(root.iterdir()):
                if p.is_dir() and (p / "config.json").exists():
                    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    out["server"].append({
                        "ref": f"server://{p.name}",
                        "label": p.name,
                        "size": size,
                    })

    # finished merge/export job outputs (scoped to caller)
    with JOBS_LOCK:
        snap = [j for j in JOBS.values()
                if j.kind in ("merge", "export") and j.status == "done"
                and (owner is None or j.owner == owner)]
    for job in sorted(snap, key=lambda j: -(j.finished_at or 0))[:20]:
        for sub in ("merged", "exported"):
            cand = ARTIFACTS_ROOT / job.id / sub
            if cand.exists() and (cand / "config.json").exists():
                size = sum(p.stat().st_size for p in cand.rglob("*") if p.is_file())
                out["jobs"].append({
                    "ref": f"job://{job.id}/{sub}",
                    "label": f"{job.kind} {job.id} -> {sub}",
                    "size": size,
                    "kind": job.kind,
                    "job_id": job.id,
                })
                break  # one entry per job

    # suggested HF ids (pure hints; the frontend lets users type any id)
    out["suggestions"] = [
        {"ref": "microsoft/codebert-base",       "label": "microsoft/codebert-base"},
        {"ref": "microsoft/graphcodebert-base",  "label": "microsoft/graphcodebert-base"},
        {"ref": "microsoft/unixcoder-base",      "label": "microsoft/unixcoder-base"},
        {"ref": "Salesforce/codet5-base",        "label": "Salesforce/codet5-base"},
    ]
    out["max_upload_bytes"] = MAX_UPLOAD_BYTES

    # -------- datasets --------
    # Bundled - read from data/benchmarks/index.json
    for b in _load_benchmarks_index().get("benchmarks", []):
        p = BENCHMARKS_ROOT / b.get("file", "")
        if not p.exists():
            continue
        out["datasets"]["bundled"].append({
            "ref": f"bundled://{b['name']}",
            "label": f"{b['name']} ({b.get('rows', '?')} rows, {b.get('balance', '?')})",
            "size": p.stat().st_size,
            "task": b.get("task"),
            "description": b.get("description", ""),
        })

    # Uploaded datasets (scoped to caller)
    if DATASET_UPLOADS_ROOT.exists():
        for d in sorted(DATASET_UPLOADS_ROOT.iterdir(),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir() or not _owns(d, owner):
                continue
            csv = d / "data.csv"
            if not csv.exists():
                continue
            label_path = d / ".label"
            label = label_path.read_text().strip() if label_path.exists() else d.name
            out["datasets"]["uploads"].append({
                "ref": f"dataset://{d.name}",
                "label": label,
                "size": csv.stat().st_size,
            })

    # Admin-mounted CSVs (MERGESE_DATASETS)
    if DATASETS_ROOT:
        droot = Path(DATASETS_ROOT)
        if droot.exists():
            for p in sorted(droot.rglob("*.csv"))[:50]:
                rel = p.relative_to(droot)
                out["datasets"]["server"].append({
                    "ref": f"server-dataset://{rel.as_posix()}",
                    "label": rel.as_posix(),
                    "size": p.stat().st_size,
                })

    # Suggested HF datasets - clickable starters for the picker
    out["datasets"]["suggestions"] = [
        {"ref": "hf-dataset://google/code_x_glue_cc_clone_detection_big_clone_bench#split=test",
         "label": "HF: code_x_glue · BigCloneBench (test)"},
        {"ref": "hf-dataset://google/code_x_glue_cc_defect_detection#split=test",
         "label": "HF: code_x_glue · Defect detection (test)"},
        {"ref": "hf-dataset://google/code_x_glue_cc_clone_detection_poj104#split=test",
         "label": "HF: code_x_glue · POJ-104 (test)"},
    ]
    return jsonify(out)


# ---- /api/datasets -----------------------------------------------------------

@app.route("/api/datasets", methods=["POST"])
def api_upload_dataset():
    """Upload a single .csv file (or .zip wrapping a .csv).

    Returns: { token, ref: "dataset://<token>", size, columns, rows_preview }
    """
    import csv as _csv

    owner = _authenticate()
    label = secure_filename((request.form.get("label") or "").strip())[:64] or None

    f = request.files.get("file") or next(iter(request.files.values()), None)
    if not f:
        return jsonify({"error": "no file"}), 400
    name = secure_filename(f.filename or "data.csv")
    if not name.lower().endswith((".csv", ".zip")):
        return jsonify({"error": "dataset must be .csv or a .zip containing one .csv"}), 400

    token = _new_upload_token()
    d = DATASET_UPLOADS_ROOT / token
    d.mkdir(parents=True, exist_ok=True)
    try:
        if name.lower().endswith(".zip"):
            zip_path = d / "_upload.zip"
            f.save(str(zip_path))
            with zipfile.ZipFile(zip_path) as zf:
                err, _ = _safe_zip_extract(zf, d)
                if err:
                    shutil.rmtree(d, ignore_errors=True)
                    return jsonify({"error": err}), 400
            zip_path.unlink(missing_ok=True)
            # Find the first CSV in the extracted tree
            csvs = list(d.rglob("*.csv"))
            if not csvs:
                shutil.rmtree(d, ignore_errors=True)
                return jsonify({"error": "no .csv found inside the zip"}), 400
            shutil.move(str(csvs[0]), str(d / "data.csv"))
            # remove the other files
            for p in d.iterdir():
                if p.name not in ("data.csv", ".label"):
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)
        else:
            f.save(str(d / "data.csv"))

        # Validate the CSV header
        with (d / "data.csv").open("r", encoding="utf-8", errors="replace") as fh:
            reader = _csv.reader(fh)
            header = next(reader, [])
            preview = []
            for i, row in enumerate(reader):
                if i >= 3: break
                preview.append({h: (v[:120] if isinstance(v, str) else v)
                                for h, v in zip(header, row)})
        cols = set(c.strip() for c in header)
        if not ({"code", "label"} <= cols) and not ({"code1", "code2", "label"} <= cols):
            shutil.rmtree(d, ignore_errors=True)
            return jsonify({
                "error": f"CSV header must include 'code,label' or 'code1,code2,label'; got {header}",
            }), 400

        if label:
            (d / ".label").write_text(label)
        _write_owner(d, owner)
        return jsonify({
            "token": token,
            "ref": f"dataset://{token}",
            "label": label or name,
            "size": (d / "data.csv").stat().st_size,
            "columns": header,
            "rows_preview": preview,
        })
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        return jsonify({"error": f"upload failed: {e}"}), 500


@app.route("/api/datasets", methods=["GET"])
def api_datasets_list():
    owner = _authenticate()
    items = []
    if DATASET_UPLOADS_ROOT.exists():
        for d in sorted(DATASET_UPLOADS_ROOT.iterdir(),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir() or not _owns(d, owner):
                continue
            csv = d / "data.csv"
            if not csv.exists():
                continue
            label_path = d / ".label"
            label = label_path.read_text().strip() if label_path.exists() else d.name
            items.append({
                "token": d.name,
                "ref": f"dataset://{d.name}",
                "label": label,
                "size": csv.stat().st_size,
            })
    return jsonify({"datasets": items, "bundled": _load_benchmarks_index().get("benchmarks", [])})


@app.route("/api/datasets/<token>", methods=["DELETE"])
def api_dataset_delete(token: str):
    owner = _authenticate()
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", token):
        return jsonify({"error": "bad token"}), 400
    d = DATASET_UPLOADS_ROOT / token
    if not d.exists() or not _owns(d, owner):
        # Return 404 (not 403) so one caller cannot probe another caller's tokens.
        return jsonify({"error": "not found"}), 404
    shutil.rmtree(d, ignore_errors=True)
    return jsonify({"ok": True})


@app.route("/api/uploads")
def api_uploads_list():
    owner = _authenticate()
    items = []
    if UPLOADS_ROOT.exists():
        for d in sorted(UPLOADS_ROOT.iterdir()):
            if not d.is_dir() or not _owns(d, owner):
                continue
            label_path = d / ".label"
            label = label_path.read_text().strip() if label_path.exists() else None
            size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
            items.append({
                "token": d.name,
                "ref": f"upload://{d.name}",
                "size": size,
                "label": label,
                "mtime": d.stat().st_mtime,
                "files": sorted(p.name for p in d.iterdir() if p.is_file())[:32],
            })
    items.sort(key=lambda x: -x["mtime"])
    return jsonify({"uploads": items, "max_upload_bytes": MAX_UPLOAD_BYTES})


@app.route("/api/uploads/<token>", methods=["DELETE"])
def api_upload_delete(token: str):
    owner = _authenticate()
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", token):
        return jsonify({"error": "bad token"}), 400
    d = _upload_dir(token)
    if not d.exists() or not _owns(d, owner):
        # Return 404 (not 403) so one caller cannot probe another caller's tokens.
        return jsonify({"error": "not found"}), 404
    shutil.rmtree(d, ignore_errors=True)
    return jsonify({"ok": True})


# ---- /api/jobs/<id>/download -------------------------------------------------

@app.route("/api/jobs/<job_id>/download")
def api_job_download(job_id: str):
    """Stream the job's output directory as a zip download."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404)
    _require_owner(job)

    # Pick the right artifact directory:
    #   merge   -> artifacts/<id>/merged
    #   export  -> artifacts/<id>/exported  (always a dir now, including the
    #              ONNX case where sibling external-data files must travel with
    #              the main .onnx file; see api_export)
    #   else    -> artifacts/<id>  (logs + result.json)
    # Legacy pre-fix exports placed .onnx/.pt at the job root; fall back to
    # those so a download of an old job still succeeds.
    job_root = ARTIFACTS_ROOT / job_id
    target: Optional[Path] = None
    if job.kind == "merge":
        target = job_root / "merged"
    elif job.kind == "export":
        for cand in ("exported", "model.onnx", "model.pt"):
            p = job_root / cand
            if p.exists():
                target = p
                break
    if target is None or not target.exists():
        target = job_root  # falls back to everything we have for the job

    archive_name = f"mergese-{job.kind}-{job_id}.zip"

    # Build the zip on disk under the job's artifact dir (reusing an existing
    # one when present), then stream the bytes back. Building it incrementally
    # in memory would corrupt the archive, since zipfile's central-directory
    # offsets don't track buffer truncation between yields.
    zip_path = ARTIFACTS_ROOT / job_id / "_download.zip"
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as zf:
            if target.is_file():
                zf.write(target, arcname=target.name)
            else:
                for f in sorted(target.rglob("*")):
                    if not f.is_file() or f.name == "_download.zip":
                        continue
                    zf.write(f, arcname=f.relative_to(target).as_posix())

    def stream_file():
        with open(zip_path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    headers = {
        "Content-Type": "application/zip",
        "Content-Disposition": f'attachment; filename="{archive_name}"',
        "Content-Length": str(zip_path.stat().st_size),
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(stream_file()), headers=headers)


# ---- background maintenance --------------------------------------------------

def _dir_newest_mtime(d: Path) -> float:
    """Most recent mtime of a directory or anything inside it."""
    newest = d.stat().st_mtime
    try:
        for f in d.rglob("*"):
            try:
                newest = max(newest, f.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return newest


def _prune_old_dirs(root: Path, ttl_hours: float, protect: Tuple[str, ...] = ()) -> int:
    """Remove immediate sub-directories of `root` untouched for longer than TTL.

    Uses the newest mtime inside each directory, so a job still writing its log
    (or an upload just referenced by a running job) is never removed.
    """
    if not root.exists():
        return 0
    cutoff = time.time() - ttl_hours * 3600.0
    removed = 0
    for d in list(root.iterdir()):
        if not d.is_dir() or d.name in protect:
            continue
        try:
            if _dir_newest_mtime(d) < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def _janitor_loop() -> None:
    while True:
        try:
            _prune_old_dirs(UPLOADS_ROOT, UPLOAD_TTL_HOURS, protect=("_datasets", "_hf_datasets"))
            _prune_old_dirs(DATASET_UPLOADS_ROOT, UPLOAD_TTL_HOURS)
            _prune_old_dirs(ARTIFACTS_ROOT, ARTIFACT_TTL_HOURS)
        except Exception:
            pass
        time.sleep(JANITOR_INTERVAL_SEC)


def _start_janitor() -> None:
    if JANITOR_ENABLED:
        threading.Thread(target=_janitor_loop, daemon=True, name="mergese-janitor").start()


_start_janitor()


# ---- auth endpoints (/api/v1) ------------------------------------------------

@app.route("/api/v1/config", methods=["GET"])
def api_public_config():
    """Public, unauthenticated: tells the front-end whether/how to authenticate."""
    return jsonify({
        "require_auth": REQUIRE_AUTH,
        "turnstile_site_key": TURNSTILE_SITE_KEY,
        "turnstile_required": bool(TURNSTILE_SECRET),
        "anon_enabled": REQUIRE_AUTH and not DISABLE_ANON,
    })


@app.route("/api/v1/anon-token", methods=["POST"])
def api_anon_token():
    """Issue a short-lived anonymous token after a bot challenge.

    When a Turnstile secret is configured the caller must pass a valid
    `cf_turnstile_response`; otherwise (dev / trusted networks) a token is
    issued freely. The emergency switch MERGESE_DISABLE_ANON refuses all of
    these without a redeploy.
    """
    if not REQUIRE_AUTH:
        return jsonify({"error": "auth is disabled on this server"}), 400
    if DISABLE_ANON:
        return jsonify({"error": "anonymous access is temporarily disabled; "
                                 "use an API key"}), 403
    if TURNSTILE_SECRET:
        import auth as _authmod
        body = request.get_json(silent=True) or {}
        tok = body.get("cf_turnstile_response") or request.form.get("cf_turnstile_response")
        if not tok or not _authmod.verify_turnstile(
                TURNSTILE_SECRET, tok, request.remote_addr):
            return jsonify({"error": "bot challenge failed"}), 403
    token, exp = _auth_store().issue_anon_token()
    return jsonify({"token": token, "expires_at": exp,
                    "token_type": "anon", "usage_header": "X-Anon-Token"})


@app.route("/api/v1/keys", methods=["POST"])
def api_mint_key():
    """Admin-only: mint an API key. Guarded by MERGESE_ADMIN_TOKEN.

    The plaintext key is returned exactly once and never stored; only its hash
    is persisted.
    """
    if not ADMIN_TOKEN or _bearer_token() != ADMIN_TOKEN:
        abort(404)  # do not advertise the admin surface
    import auth as _authmod
    body = request.get_json(silent=True) or {}
    tier = body.get("tier", "key")
    email = body.get("email")
    daily_limit = body.get("daily_limit")
    try:
        client_id, plaintext = _auth_store().mint_key(email, tier, daily_limit)
    except _authmod.AuthError as e:
        return jsonify({"error": e.message}), e.status
    return jsonify({"client_id": client_id, "api_key": plaintext, "tier": tier,
                    "note": "store this key now; it will not be shown again"}), 201


@app.route("/api/v1/keys/<client_id>", methods=["DELETE"])
def api_revoke_key(client_id: str):
    if not ADMIN_TOKEN or _bearer_token() != ADMIN_TOKEN:
        abort(404)
    ok = _auth_store().revoke_key(client_id)
    return jsonify({"revoked": ok}), (200 if ok else 404)


@app.route("/api/v1/limits", methods=["GET"])
def api_limits():
    """Report the caller's quota and current usage."""
    if not REQUIRE_AUTH:
        return jsonify({"auth": "disabled",
                        "note": "this server accepts unauthenticated requests"})
    client = _resolve_client()
    usage = _auth_store().usage(client)
    usage["active_jobs"] = _client_active_jobs(client.client_id)
    return jsonify(usage)


# ---- error handlers ----------------------------------------------------------

@app.errorhandler(400)
def _bad_request(e):
    """Make `abort(400, description=...)` return JSON so API clients can parse
    the error. Without this handler the default is Werkzeug's HTML page."""
    return jsonify({"error": getattr(e, "description", "bad request")}), 400


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "upload too large",
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "hint": "Increase MERGESE_MAX_UPLOAD_BYTES on the server, or split the upload.",
    }), 413


def _register_auth_error_handler():
    import auth as _authmod

    @app.errorhandler(_authmod.AuthError)
    def _auth_error(e):  # noqa: ANN001
        return jsonify({"error": e.message}), e.status


_register_auth_error_handler()


# ---- main --------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("MERGESE_PORT", "8765"))
    host = os.environ.get("MERGESE_HOST", "0.0.0.0")
    debug = bool(int(os.environ.get("MERGESE_DEBUG", "0")))
    app.run(host=host, port=port, debug=debug, threaded=True)
