#!/usr/bin/env bash
# Provision a fresh Ubuntu VM to serve the MergeSE web tool.
#
# Idempotent: safe to re-run, for example to add TLS once DNS points at the VM.
# Installs Docker, builds and starts the container, configures nginx as a
# reverse proxy, and opens the firewall. With --domain it also obtains a
# Let's Encrypt certificate.
#
# Usage:
#   sudo ./provision.sh
#   sudo ./provision.sh --domain mergese.usask.ca --email you@usask.ca
#
# Options:
#   --domain <fqdn>   Serve this hostname and request a TLS certificate.
#   --email  <addr>   Contact address for Let's Encrypt (required with --domain).
#   --repo   <url>    Git URL to deploy (default: the public MergeSE repository).
#   --dir    <path>   Install directory (default: /opt/mergese).

set -euo pipefail

REPO="https://github.com/srlabUsask/MergeSE.git"
DIR="/opt/mergese"
DOMAIN=""
EMAIL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --domain) DOMAIN="$2"; shift 2 ;;
        --email)  EMAIL="$2";  shift 2 ;;
        --repo)   REPO="$2";   shift 2 ;;
        --dir)    DIR="$2";    shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "This script must run as root (use sudo)." >&2
    exit 1
fi

if [ -n "$DOMAIN" ] && [ -z "$EMAIL" ]; then
    echo "--domain requires --email for Let's Encrypt registration." >&2
    exit 2
fi

export DEBIAN_FRONTEND=noninteractive

echo "[1/6] Installing base packages"
apt-get update -y
apt-get install -y ca-certificates curl git ufw nginx

echo "[2/6] Installing Docker Engine"
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

echo "[3/6] Configuring firewall (SSH, HTTP, HTTPS)"
ufw allow OpenSSH || true
ufw allow 80/tcp || true
ufw allow 443/tcp || true
ufw --force enable

echo "[4/6] Fetching MergeSE into ${DIR}"
if [ -d "${DIR}/.git" ]; then
    git -C "${DIR}" pull --ff-only
else
    git clone "${REPO}" "${DIR}"
fi

echo "[5/6] Building and starting the container"
cd "${DIR}"
docker compose up -d --build

echo "[6/6] Configuring nginx reverse proxy"
SERVER_NAME="${DOMAIN:-_}"
cat > /etc/nginx/sites-available/mergese <<NGINX
# Per-client-IP rate limiting to blunt request floods.
limit_req_zone  \$binary_remote_addr zone=mergese_general:10m rate=10r/s;
limit_req_zone  \$binary_remote_addr zone=mergese_jobs:10m    rate=1r/s;
limit_conn_zone \$binary_remote_addr zone=mergese_conn:10m;

server {
    listen 80;
    server_name ${SERVER_NAME};

    client_max_body_size 3g;
    proxy_read_timeout   3600s;
    proxy_request_buffering off;

    limit_conn mergese_conn 20;

    # Server-Sent-Events log stream: buffering must be off.
    location ~ ^/api/jobs/[^/]+/stream\$ {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        chunked_transfer_encoding on;
        proxy_read_timeout 24h;
        add_header X-Accel-Buffering no;
    }

    # Job-starting and upload endpoints: tighter rate limit.
    location ~ ^/api/(inspect|merge|evaluate|export|uploads|datasets)\$ {
        limit_req zone=mergese_jobs burst=5 nodelay;
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location / {
        limit_req zone=mergese_general burst=20 nodelay;
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/mergese /etc/nginx/sites-enabled/mergese
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

if [ -n "${DOMAIN}" ]; then
    echo "Requesting TLS certificate for ${DOMAIN}"
    apt-get install -y certbot python3-certbot-nginx
    certbot --nginx -d "${DOMAIN}" --non-interactive --agree-tos -m "${EMAIL}" --redirect
fi

echo
echo "MergeSE is running."
if [ -n "${DOMAIN}" ]; then
    echo "  https://${DOMAIN}"
else
    echo "  http://<this-vm-ip>/"
    echo "  Add HTTPS later: sudo ${DIR}/deploy/provision.sh --domain <fqdn> --email <addr>"
fi
