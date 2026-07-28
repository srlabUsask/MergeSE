"""Authentication and per-caller quotas for the MergeSE server.

Design goals (see docs/SECURITY.md):

  * No user profiles, no passwords. Every caller carries an identity - either a
    long-lived **API key** (CLI / programmatic use) or a short-lived signed
    **anonymous token** (website use, gated by a bot challenge).
  * Keys are stored only as SHA-256 hashes; the plaintext is shown once.
  * Quotas are enforced per identity: a daily job budget and a cap on
    concurrently active jobs. Job ownership is recorded so one caller can never
    read, cancel, or download another's job.
  * Everything is OFF by default (`MERGESE_REQUIRE_AUTH=0`) so a trusted
    single-tenant deployment keeps working unchanged. Flip it on for public
    exposure.

This module is intentionally free of Flask imports so it can be unit-tested in
isolation; the web layer adapts it with a small decorator.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---- tiers -------------------------------------------------------------------

# Each tier caps the daily job budget and the number of simultaneously active
# (pending/running) jobs. Deliberately conservative defaults; an operator can
# widen the "approved" tier per collaborator by minting a key with a custom
# daily limit.
@dataclass(frozen=True)
class Tier:
    name: str
    daily_jobs: int
    max_active: int


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Per-tier caps. Defaults comfortably fit several full inspect -> merge ->
# evaluate -> export workflows a day; every value is overridable by env so an
# operator can tune quotas without a code change.
TIERS: Dict[str, Tier] = {
    "anonymous": Tier("anonymous",
                      daily_jobs=_int_env("MERGESE_ANON_DAILY_JOBS", 30),
                      max_active=_int_env("MERGESE_ANON_MAX_ACTIVE", 2)),
    "key": Tier("key",
                daily_jobs=_int_env("MERGESE_KEY_DAILY_JOBS", 100),
                max_active=_int_env("MERGESE_KEY_MAX_ACTIVE", 3)),
    "approved": Tier("approved",
                     daily_jobs=_int_env("MERGESE_APPROVED_DAILY_JOBS", 1000),
                     max_active=_int_env("MERGESE_APPROVED_MAX_ACTIVE", 5)),
}

KEY_PREFIX = "mse_live_"


def _now() -> float:
    return time.time()


def _day_stamp(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts if ts is not None else _now()))


def hash_key(plaintext: str) -> str:
    """SHA-256 of an API key. Only the hash is ever persisted."""
    return hashlib.sha256(plaintext.encode()).hexdigest()


# ---- identity resolved for a request -----------------------------------------

@dataclass
class Client:
    client_id: str
    kind: str           # "key" | "anonymous"
    tier: Tier
    daily_limit: int    # effective per-day budget (tier default or per-key override)


class AuthError(Exception):
    """Raised with an HTTP-ish status for the web layer to translate."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class AuthStore:
    """SQLite-backed API-key registry, anon-token signer, and quota ledger."""

    def __init__(self, db_path: Path, secret: bytes,
                 anon_ttl_sec: int = 3600):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._secret = secret
        self.anon_ttl_sec = anon_ttl_sec
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_clients (
                    client_id   TEXT PRIMARY KEY,
                    key_hash    TEXT UNIQUE NOT NULL,
                    email       TEXT,
                    tier        TEXT NOT NULL DEFAULT 'key',
                    daily_limit INTEGER,
                    created_at  REAL NOT NULL,
                    last_used_at REAL,
                    disabled    INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS usage_ledger (
                    client_id TEXT NOT NULL,
                    day       TEXT NOT NULL,
                    jobs_used INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (client_id, day)
                );
                """
            )

    # ---- API keys -----------------------------------------------------------

    def mint_key(self, email: Optional[str] = None, tier: str = "key",
                 daily_limit: Optional[int] = None) -> Tuple[str, str]:
        """Create a key. Returns (client_id, plaintext_key). Store only the hash."""
        if tier not in TIERS:
            raise AuthError(400, f"unknown tier {tier!r}")
        client_id = "cli_" + secrets.token_hex(8)
        plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO api_clients "
                "(client_id, key_hash, email, tier, daily_limit, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (client_id, hash_key(plaintext), email, tier, daily_limit, _now()),
            )
        return client_id, plaintext

    def revoke_key(self, client_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE api_clients SET disabled=1 WHERE client_id=?", (client_id,))
            return cur.rowcount > 0

    def _client_from_key(self, plaintext: str) -> Client:
        row = self._conn.execute(
            "SELECT * FROM api_clients WHERE key_hash=?",
            (hash_key(plaintext),)).fetchone()
        if row is None:
            raise AuthError(401, "invalid API key")
        if row["disabled"]:
            raise AuthError(403, "API key has been revoked")
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE api_clients SET last_used_at=? WHERE client_id=?",
                (_now(), row["client_id"]))
        tier = TIERS.get(row["tier"], TIERS["key"])
        limit = row["daily_limit"] if row["daily_limit"] is not None else tier.daily_jobs
        return Client(row["client_id"], "key", tier, int(limit))

    # ---- anonymous tokens (stateless, HMAC-signed) --------------------------

    def issue_anon_token(self) -> Tuple[str, int]:
        """Return (token, expiry_epoch). Single job budget, short TTL.

        Stateless: the token embeds its own client id and expiry, signed with
        the server secret, so no storage is needed to validate it. The daily
        quota is still enforced through the usage ledger keyed by that id.
        """
        exp = int(_now()) + self.anon_ttl_sec
        cid = "anon_" + secrets.token_hex(8)
        payload = f"{cid}.{exp}".encode()
        sig = hmac.new(self._secret, payload, hashlib.sha256).digest()
        # payload is appended to a fixed-length (32-byte) signature with no
        # delimiter. A delimiter byte can also occur inside the raw HMAC digest,
        # which made an earlier rsplit-based parser fail ~12% of the time.
        token = base64.urlsafe_b64encode(payload + sig).decode().rstrip("=")
        return token, exp

    def _client_from_anon(self, token: str) -> Client:
        try:
            raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
            # SHA-256 HMAC is always exactly 32 bytes: slice it off the end
            # rather than splitting on a delimiter that can appear in the digest.
            if len(raw) < 33:  # >=1 byte of payload + 32-byte signature
                raise AuthError(401, "malformed anonymous token")
            payload, sig = raw[:-32], raw[-32:]
            expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(sig, expected):
                raise AuthError(401, "invalid anonymous token")
            cid_b, exp_b = payload.split(b".", 1)
            cid, exp = cid_b.decode(), int(exp_b)
        except AuthError:
            raise
        except Exception:
            raise AuthError(401, "malformed anonymous token")
        if _now() > exp:
            raise AuthError(401, "anonymous token has expired")
        tier = TIERS["anonymous"]
        return Client(cid, "anonymous", tier, tier.daily_jobs)

    # ---- request-time resolution -------------------------------------------

    def authenticate(self, api_key: Optional[str],
                     anon_token: Optional[str]) -> Client:
        """Resolve a caller from an API key or an anonymous token."""
        if api_key:
            return self._client_from_key(api_key)
        if anon_token:
            return self._client_from_anon(anon_token)
        raise AuthError(401, "authentication required: supply an API key or "
                             "obtain an anonymous token")

    # ---- quota ledger -------------------------------------------------------

    def _jobs_used_today(self, client_id: str) -> int:
        row = self._conn.execute(
            "SELECT jobs_used FROM usage_ledger WHERE client_id=? AND day=?",
            (client_id, _day_stamp())).fetchone()
        return int(row["jobs_used"]) if row else 0

    def check_and_reserve(self, client: Client, active_jobs: int) -> None:
        """Enforce active-job and daily-job limits, then debit the daily budget.

        `active_jobs` is the caller's current pending/running count, supplied by
        the web layer (which owns the job table). Raises AuthError(429) when a
        limit is hit; otherwise records one job against today's budget.
        """
        if active_jobs >= client.tier.max_active:
            raise AuthError(429, f"you already have {active_jobs} active job(s); "
                                 f"limit is {client.tier.max_active}")
        with self._lock, self._conn:
            used = self._jobs_used_today(client.client_id)
            if used >= client.daily_limit:
                raise AuthError(429, f"daily job limit reached "
                                     f"({used}/{client.daily_limit})")
            self._conn.execute(
                "INSERT INTO usage_ledger (client_id, day, jobs_used) VALUES (?,?,1) "
                "ON CONFLICT(client_id, day) DO UPDATE SET jobs_used = jobs_used + 1",
                (client.client_id, _day_stamp()))

    def usage(self, client: Client) -> Dict[str, object]:
        with self._lock:
            used = self._jobs_used_today(client.client_id)
        return {
            "client_id": client.client_id,
            "kind": client.kind,
            "tier": client.tier.name,
            "daily_limit": client.daily_limit,
            "jobs_used_today": used,
            "jobs_remaining_today": max(0, client.daily_limit - used),
            "max_active_jobs": client.tier.max_active,
        }


# ---- Cloudflare Turnstile verification (optional) ----------------------------

def verify_turnstile(secret: str, response_token: str,
                     remote_ip: Optional[str] = None, timeout: float = 5.0) -> bool:
    """Server-side validation of a Turnstile challenge response.

    Returns True on success. Kept dependency-free (urllib) and defensive: any
    network or parse error is a verification failure, never an exception that
    would fault the request handler.
    """
    import json
    import urllib.parse
    import urllib.request

    data = {"secret": secret, "response": response_token}
    if remote_ip:
        data["remoteip"] = remote_ip
    try:
        req = urllib.request.Request(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data=urllib.parse.urlencode(data).encode(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode())
        return bool(body.get("success"))
    except Exception:
        return False


def load_secret(explicit: Optional[str], path: Path) -> bytes:
    """Return a stable signing secret.

    Precedence: an explicit env secret, then a persisted random secret, then a
    freshly generated one written to `path` (mode 0600). Persisting it means
    anonymous tokens survive a restart.
    """
    if explicit:
        return explicit.encode()
    try:
        if path.exists():
            return path.read_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        secret = secrets.token_bytes(32)
        path.write_bytes(secret)
        os.chmod(path, 0o600)
        return secret
    except OSError:
        # Fall back to an ephemeral secret (anon tokens won't survive restart).
        return secrets.token_bytes(32)
