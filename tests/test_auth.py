"""Tests for the auth store: API keys, anonymous tokens, quotas, and ownership.

These exercise server/auth.py directly - no Flask, no network.
"""
import importlib.util
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_auth():
    spec = importlib.util.spec_from_file_location("mergese_auth", ROOT / "server" / "auth.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mergese_auth"] = mod
    spec.loader.exec_module(mod)
    return mod


auth = _load_auth()


@pytest.fixture
def store(tmp_path):
    return auth.AuthStore(tmp_path / "auth.db", b"test-secret-key", anon_ttl_sec=3600)


# ---- API keys ----------------------------------------------------------------

def test_mint_and_authenticate_key(store):
    cid, key = store.mint_key(email="a@b.c")
    assert key.startswith(auth.KEY_PREFIX)
    client = store.authenticate(api_key=key, anon_token=None)
    assert client.client_id == cid
    assert client.kind == "key"


def test_only_hash_is_stored(store):
    _, key = store.mint_key()
    row = store._conn.execute("SELECT key_hash FROM api_clients").fetchone()
    assert row["key_hash"] == auth.hash_key(key)
    assert key not in row["key_hash"]


def test_invalid_key_rejected(store):
    with pytest.raises(auth.AuthError) as ei:
        store.authenticate(api_key="mse_live_nope", anon_token=None)
    assert ei.value.status == 401


def test_revoked_key_rejected(store):
    cid, key = store.mint_key()
    assert store.revoke_key(cid) is True
    with pytest.raises(auth.AuthError) as ei:
        store.authenticate(api_key=key, anon_token=None)
    assert ei.value.status == 403


def test_custom_daily_limit_overrides_tier(store):
    _, key = store.mint_key(tier="key", daily_limit=3)
    client = store.authenticate(api_key=key, anon_token=None)
    assert client.daily_limit == 3


# ---- anonymous tokens --------------------------------------------------------

def test_anon_token_roundtrip(store):
    token, exp = store.issue_anon_token()
    assert exp > time.time()
    client = store.authenticate(api_key=None, anon_token=token)
    assert client.kind == "anonymous"


def test_tampered_anon_token_rejected(store):
    token, _ = store.issue_anon_token()
    bad = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    with pytest.raises(auth.AuthError):
        store.authenticate(api_key=None, anon_token=bad)


def test_expired_anon_token_rejected(tmp_path):
    st = auth.AuthStore(tmp_path / "a.db", b"s", anon_ttl_sec=-1)  # already expired
    token, _ = st.issue_anon_token()
    with pytest.raises(auth.AuthError) as ei:
        st.authenticate(api_key=None, anon_token=token)
    assert ei.value.status == 401


def test_missing_credentials_rejected(store):
    with pytest.raises(auth.AuthError) as ei:
        store.authenticate(api_key=None, anon_token=None)
    assert ei.value.status == 401


# ---- quotas ------------------------------------------------------------------

def test_daily_job_limit_enforced(store):
    _, key = store.mint_key(daily_limit=2)
    client = store.authenticate(api_key=key, anon_token=None)
    store.check_and_reserve(client, active_jobs=0)  # 1
    store.check_and_reserve(client, active_jobs=0)  # 2
    with pytest.raises(auth.AuthError) as ei:
        store.check_and_reserve(client, active_jobs=0)  # 3 -> over
    assert ei.value.status == 429
    assert "daily" in ei.value.message.lower()


def test_active_job_limit_enforced(store):
    _, key = store.mint_key()
    client = store.authenticate(api_key=key, anon_token=None)
    # Simulate the caller already at their tier's active-job cap.
    cap = client.tier.max_active
    with pytest.raises(auth.AuthError) as ei:
        store.check_and_reserve(client, active_jobs=cap)
    assert ei.value.status == 429
    assert "active" in ei.value.message.lower()


def test_usage_report(store):
    _, key = store.mint_key(daily_limit=5)
    client = store.authenticate(api_key=key, anon_token=None)
    store.check_and_reserve(client, active_jobs=0)
    u = store.usage(client)
    assert u["jobs_used_today"] == 1
    assert u["jobs_remaining_today"] == 4
    assert u["tier"] == "key"


def test_quota_independent_per_client(store):
    _, k1 = store.mint_key(daily_limit=1)
    _, k2 = store.mint_key(daily_limit=1)
    c1 = store.authenticate(api_key=k1, anon_token=None)
    c2 = store.authenticate(api_key=k2, anon_token=None)
    store.check_and_reserve(c1, active_jobs=0)
    # c1 is now exhausted, but c2 is untouched.
    store.check_and_reserve(c2, active_jobs=0)
    with pytest.raises(auth.AuthError):
        store.check_and_reserve(c1, active_jobs=0)
