# MergeSE security & deployment hardening

MergeSE accepts externally-supplied model files and runs compute jobs on them.
This document describes the protections built into the tool, how to turn them
on, and what still has to be done at the deployment / university-IT layer before
exposing it publicly.

Everything here is **off by default** so a trusted single-tenant install keeps
working unchanged. Turn the switches on for any publicly reachable deployment.

---

## 1. What the tool enforces (in code)

### Untrusted model files never execute code
* Uploads are **safetensors-only** by default. Pickle-backed checkpoints
  (`.bin`, `.pt`, `.ckpt`, `.pkl`, ...) and code files (`.py`, `.so`, ...) are
  refused at upload time (`_scan_unsafe_files`). Pickle checkpoints run
  arbitrary code on load; safetensors are tensor-only.
* Every `torch.load` uses `weights_only=True`, so even a pickle that reaches the
  loader cannot execute code.
* Override only if you fully trust your users: `MERGESE_ALLOW_PICKLE_UPLOADS=1`.

### Archive uploads are bounded
`_safe_zip_extract` rejects: path traversal / absolute paths (Zip Slip), too
many entries, an oversized uncompressed total (checked from the header *and*
enforced during the copy), and pathological per-entry compression ratios (zip
bombs). Tunable via `MERGESE_MAX_UNCOMPRESSED_BYTES`, `MERGESE_MAX_ARCHIVE_ENTRIES`,
`MERGESE_MAX_COMPRESSION_RATIO`.

### The merge worker is sandboxed
Each job runs in a hardened child process (`_build_worker`):
* **Stripped environment** — only `PATH`, a job-private `HOME`/`TMPDIR`, and the
  read-only HF cache path. The full server environment (secrets, DB URLs) is
  never inherited.
* **Job-private working directory** — `artifacts/<id>/`, never the app source.
* **No network** — when the host supports an unprivileged network namespace the
  worker is wrapped in `unshare -rn`, so it cannot reach the internet, the Flask
  service, a database, localhost, or cloud-metadata endpoints. This removes the
  SSRF / network-probing surface entirely.
* **Resource limits (rlimits)** — CPU seconds (`MERGESE_WORKER_CPU_SEC`), max
  output file size (`MERGESE_WORKER_MAX_FILE_BYTES`), optional address-space cap.
* **Hard wall-clock timeout** — `MERGESE_WORKER_TIMEOUT_SEC` (default 1h); on
  expiry the whole process group is killed.

Because the worker is offline, HuggingFace models must be **pre-cached** on the
server (or uploaded as safetensors). A job referencing an uncached Hub id is
refused with an actionable message rather than silently fetching it. Set
`MERGESE_WORKER_OFFLINE=0` only if you accept on-demand Hub fetches from the
worker.

> **Note on memory / PID limits.** Reliable per-job RAM and process-count caps
> need cgroups (`memory.max`, `pids.max`), which require privilege this tool
> does not assume. `RLIMIT_NPROC` is **off by default** because on Linux it
> counts every process owned by the uid, not just the job, and trips torch on a
> shared host. On a single-tenant box you can set `MERGESE_WORKER_MAX_NPROC`.
> For true multi-tenant isolation, run the worker under a container/VM with
> cgroup limits (see §4).

### Command execution is injection-free
The Flask layer never builds shell strings. Jobs are launched with a list of
arguments and `shell=False`; every user-controllable option is an enum, a
bounded number, or a server-generated path validated to stay inside its root
(`resolve_model_ref`). Users cannot pass raw CLI strings.

---

## 2. Authentication & quotas (opt-in)

Set `MERGESE_REQUIRE_AUTH=1` to require an identity on every mutating request
(`merge`, `evaluate`, `export`, `upload`). Two credential types:

| Caller | Credential | How obtained |
|--------|-----------|--------------|
| Website user | short-lived **anonymous token** | `POST /api/v1/anon-token` after a Turnstile challenge |
| CLI / programmatic | long-lived **API key** | minted by an admin |

Keys are stored only as SHA-256 hashes; the plaintext is shown once. Tiers cap a
daily job budget and simultaneous active jobs (`anonymous` / `key` / `approved`,
see `server/auth.py`). Job **ownership** is recorded, so one caller can never
read, cancel, or download another's job (cross-tenant requests return 404, not
403, so job existence isn't leaked).

### Config

| Env var | Meaning |
|---------|---------|
| `MERGESE_REQUIRE_AUTH` | `1` to require auth (default `0`) |
| `MERGESE_ADMIN_TOKEN` | bearer token that guards key minting (`/api/v1/keys`) |
| `MERGESE_TURNSTILE_SECRET` | Cloudflare Turnstile secret; when set, anon tokens require a passing challenge |
| `MERGESE_DISABLE_ANON` | **emergency switch**: refuse anonymous tokens, require API keys, no redeploy |
| `MERGESE_AUTH_DB` | sqlite path for the key/quota store |
| `MERGESE_AUTH_SECRET` | signing secret for anon tokens (persisted if unset) |
| `MERGESE_ANON_TTL_SEC` | anon-token lifetime (default 3600) |

### Minting keys

Online (admin token required):
```bash
curl -X POST -H "Authorization: Bearer $MERGESE_ADMIN_TOKEN" \
     -H 'content-type: application/json' \
     -d '{"email":"alice@uni.edu","tier":"key"}' \
     https://<host>/api/v1/keys
```

Offline (no server needed):
```bash
python server/manage_keys.py mint --email alice@uni.edu --tier key
python server/manage_keys.py list
python server/manage_keys.py revoke cli_1a2b3c4d5e6f7a8b
```

### Using a key / token

```bash
# API key
curl -H "Authorization: Bearer mse_live_..." ...
# anonymous token
curl -H "X-Anon-Token: <token>" ...
# check your quota
curl -H "Authorization: Bearer mse_live_..." https://<host>/api/v1/limits
```

---

## 3. What is NOT in the tool (do this at the deployment layer)

The application cannot provide these; configure them around it:

* **TLS + production WSGI** — run behind nginx/university ingress → gunicorn.
  Never expose `app.run()` / Flask debug publicly.
* **HTTP rate limiting** — per-IP request rate, connection caps, request
  timeouts, and body-size limits belong in nginx (or Flask-Limiter with Redis).
  The tool enforces *job* quotas; it does not throttle raw HTTP.
* **Security headers / CORS / host validation** — set CSP, HSTS,
  `X-Content-Type-Options`, a non-`*` CORS policy, and host-header validation at
  the proxy.
* **Secrets management** — inject `MERGESE_ADMIN_TOKEN`, `MERGESE_TURNSTILE_SECRET`,
  etc. from the university secret manager, not from committed files.
* **Central logging** off-host, vulnerability scanning, and an incident contact.

---

## 4. Stronger isolation (multi-tenant / GPU)

The built-in same-host sandbox (stripped env + offline netns + rlimits) is solid
for a low-trust single box. For untrusted multi-tenant use, run each job in a
disposable **container or VM** with cgroup memory/PID limits, a read-only root
filesystem, dropped capabilities, `no-new-privileges`, a seccomp profile, and
`--network none`. MergeSE already runs the worker as an isolated subprocess, so
swapping `_build_worker` to launch that subprocess *inside* a per-job container
is a localized change.

---

## 5. Minimum go-live checklist

Before exposing MergeSE publicly:

- [ ] University IT has approved the architecture and placement (DMZ/VLAN).
- [ ] `MERGESE_REQUIRE_AUTH=1`, with a strong `MERGESE_ADMIN_TOKEN`.
- [ ] `MERGESE_TURNSTILE_SECRET` set (or anonymous access disabled).
- [ ] Safetensors-only uploads (default; `MERGESE_ALLOW_PICKLE_UPLOADS` unset).
- [ ] Worker offline confirmed (`unshare -rn` available, or a container with
      `--network none`).
- [ ] TLS + gunicorn + nginx with rate limits, body caps, and security headers.
- [ ] Base encoders pre-cached; arbitrary Hub fetches disabled.
- [ ] Off-host logging + an emergency `MERGESE_DISABLE_ANON` runbook.
- [ ] An independent security review / pen test.
