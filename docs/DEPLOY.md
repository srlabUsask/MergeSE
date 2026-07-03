# Deploying MergeSE on a VM

This guide stands up the MergeSE web tool on a dedicated Ubuntu VM (for
example, an OpenStack instance on a university cloud). The tool runs as a
Docker container behind an nginx reverse proxy, with optional Let's Encrypt
TLS.

## VM sizing

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| vCPU     | 2       | 4+          |
| RAM      | 4 GB    | 8 GB        |
| Disk     | 40 GB   | 80 GB+      |
| OS       | Ubuntu 22.04 LTS | Ubuntu 22.04 / 24.04 LTS |

Merging is light. Evaluation is CPU-bound and benefits from more cores; a GPU
is optional and only speeds up evaluation and WUDI merges. Disk is dominated by
the HuggingFace model cache and uploaded checkpoints, so size it to the models
you expect to host.

The VM needs outbound internet access to pull base images from the HuggingFace
Hub at runtime.

## 1. Launch the VM (OpenStack)

1. Choose an Ubuntu 22.04 LTS image and a flavor meeting the sizing above.
2. Add an SSH key pair so you can log in.
3. Create or select a security group that allows inbound:
   - TCP 22 (SSH), restricted to your admin network if possible
   - TCP 80 (HTTP)
   - TCP 443 (HTTPS)
4. (Optional, hands-off) Paste `deploy/cloud-init.yaml` as the instance user
   data. The VM will install everything and start MergeSE on port 80 during
   first boot.
5. Assign a floating IP so the VM is reachable from outside.

## 2. Provision (if you did not use cloud-init)

SSH into the VM and run the provisioning script:

```bash
sudo apt-get update && sudo apt-get install -y git
sudo git clone https://github.com/srlabUsask/MergeSE.git /opt/mergese
sudo /opt/mergese/deploy/provision.sh
```

This installs Docker, builds and starts the container, configures nginx, and
opens the firewall. The tool is now reachable at `http://<floating-ip>/`.

## 3. DNS and TLS

1. Create a DNS A record for your hostname (for example `mergese.usask.ca`)
   pointing at the VM's floating IP. Wait for it to resolve.
2. Obtain a certificate and enable HTTPS:

```bash
sudo /opt/mergese/deploy/provision.sh --domain mergese.usask.ca --email you@usask.ca
```

The script requests a Let's Encrypt certificate, rewrites the nginx site to
serve HTTPS, and redirects HTTP to HTTPS. Certbot installs a systemd timer that
renews the certificate automatically.

## 4. Verify

```bash
curl -fsS http://localhost:8765/api/health          # from the VM
curl -fsS https://mergese.usask.ca/api/health        # once TLS is live
```

Both should return `{"ok": true, ...}`. Then open the site in a browser.

## Operations

All commands run from the install directory (`/opt/mergese`).

```bash
# View logs
docker compose logs -f

# Restart
docker compose restart

# Update to the latest release
git pull --ff-only && docker compose up -d --build

# Stop
docker compose down
```

State lives in Docker named volumes (`hf_cache`, `artifacts`, `uploads`). Back
these up if you need to preserve uploaded checkpoints and merge outputs. They
survive `docker compose down` and are only removed by `docker compose down -v`.

## Configuration

The server reads these environment variables (set them in `docker-compose.yml`
under the `environment:` block, or in the systemd unit):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MERGESE_MAX_CONCURRENT`     | `2`    | Maximum simultaneous merge/evaluate jobs. |
| `MERGESE_MAX_QUEUE`          | `8`    | Queued + running jobs allowed before new submissions get a 429. |
| `MERGESE_MAX_UPLOAD_BYTES`   | `3 GB` | Upload size cap. |
| `MERGESE_UPLOAD_TTL_HOURS`   | `48`   | Age at which uploads and uploaded datasets are auto-removed. |
| `MERGESE_ARTIFACT_TTL_HOURS` | `48`   | Age at which job artifacts (merge outputs, logs) are auto-removed. |
| `MERGESE_CHECKPOINTS`        | unset  | Optional directory of admin-mounted checkpoints. |
| `MERGESE_ALLOW_LOCAL_PATHS`  | `0`    | Keep `0` on a public server; `1` lets visitors reference host paths. |

## Security and abuse resistance

The tool is designed to be publicly usable without a login, so the protections
are layered rather than relying on authentication.

- **Volumetric DDoS:** put a CDN in front (Cloudflare's free tier or your
  institution's CDN). This is the most effective defence against large floods
  and is strongly recommended for a public deployment. The nginx config works
  behind a proxy unchanged.
- **Request floods:** nginx applies per-IP rate limits (10 req/s general, 1
  req/s for job-starting endpoints) and a per-IP connection cap.
- **Compute exhaustion:** at most `MERGESE_MAX_CONCURRENT` jobs run at once, and
  submissions beyond `MERGESE_MAX_QUEUE` are rejected with HTTP 429, so a flood
  of slow merges cannot pile up.
- **Disk exhaustion:** a background janitor removes uploads and job artifacts
  older than their TTL, so repeated large uploads cannot fill the disk.
- **Host containment:** the container has CPU, memory, and PID limits in
  `docker-compose.yml`, so one heavy or runaway job cannot take down the VM.
- **Input safety:** `MERGESE_ALLOW_LOCAL_PATHS` is `0` by default, so visitors
  can only use uploads or HuggingFace Hub IDs, never arbitrary host paths;
  uploads are size-capped and extracted with Zip-Slip protection.
- **Network exposure:** the container binds to `127.0.0.1:8765`, so only nginx
  is reachable publicly. Restrict SSH (port 22) to trusted networks in the
  security group.

Tune the rate limits, TTLs, and resource caps to your VM size and expected
traffic. For a small VM, lower `MERGESE_MAX_CONCURRENT` and the `cpus`/`mem_limit`
values in `docker-compose.yml`.

## Bare-metal alternative

To run without Docker, use the systemd unit at `deploy/mergese.service` with a
Python virtual environment and gunicorn. See the comments in that file and in
`deploy/nginx.conf`.
