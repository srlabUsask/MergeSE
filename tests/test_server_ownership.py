"""Ownership and scoping tests for the web server.

These exercise the per-caller isolation of uploads, dataset uploads, job
artifacts, and the resolve_model_ref / resolve_dataset_ref gates - without
needing a running Flask server or any real checkpoints. They also pin the
preset catalogue to refs the server will actually accept.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


def _load_app(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("MERGESE_UPLOADS", str(tmp_path / "uploads"))
    monkeypatch.setenv("MERGESE_ARTIFACTS", str(tmp_path / "artifacts"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("mergese_app", ROOT / "server" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mergese_app"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_hf_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00" * 16)


# ---- owner helpers -----------------------------------------------------------

def test_owner_roundtrip(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    d = tmp_path / "x"
    d.mkdir()
    app._write_owner(d, "alice")
    assert app._read_owner(d) == "alice"


def test_owns_when_auth_off(monkeypatch, tmp_path):
    """When owner=None (auth disabled), every dir is accessible."""
    app = _load_app(monkeypatch, tmp_path)
    d = tmp_path / "x"
    d.mkdir()
    # No owner file, auth off -> accessible
    assert app._owns(d, None) is True
    app._write_owner(d, "alice")
    assert app._owns(d, None) is True  # still accessible when auth off


def test_owns_requires_match_when_auth_on(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    d = tmp_path / "x"
    d.mkdir()
    # No owner file at all - a public deployment should NOT grant access.
    assert app._owns(d, "alice") is False
    app._write_owner(d, "alice")
    assert app._owns(d, "alice") is True
    assert app._owns(d, "bob") is False


# ---- resolve_model_ref ownership --------------------------------------------

def test_resolve_upload_refuses_other_owner(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    token = "abc123def456"
    d = app.UPLOADS_ROOT / token
    _make_hf_dir(d)
    app._write_owner(d, "alice")

    # Owner can resolve the ref.
    assert app.resolve_model_ref(f"upload://{token}", owner="alice") == str(d.resolve())

    # Non-owner gets a "not found" error that does not leak existence.
    with pytest.raises(ValueError) as ei:
        app.resolve_model_ref(f"upload://{token}", owner="bob")
    assert "not found" in str(ei.value).lower()


def test_resolve_upload_allows_everyone_when_auth_off(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    token = "abc123def456"
    d = app.UPLOADS_ROOT / token
    _make_hf_dir(d)
    app._write_owner(d, "alice")

    # owner=None (auth disabled / trusted single-tenant mode) -> accessible.
    assert app.resolve_model_ref(f"upload://{token}") == str(d.resolve())


def test_resolve_job_ref_refuses_other_owner(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    jid = "abcdef123456"
    root = app.ARTIFACTS_ROOT / jid
    merged = root / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}")
    app._write_owner(root, "alice")

    assert app.resolve_model_ref(f"job://{jid}", owner="alice") == str(merged.resolve())
    with pytest.raises(ValueError) as ei:
        app.resolve_model_ref(f"job://{jid}", owner="bob")
    assert "not found" in str(ei.value).lower()


def test_resolve_dataset_refuses_other_owner(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    token = "ds_abcdef12"
    d = app.DATASET_UPLOADS_ROOT / token
    d.mkdir(parents=True)
    (d / "data.csv").write_text("code,label\nfoo,1\n")
    app._write_owner(d, "alice")

    assert app.resolve_dataset_ref(f"dataset://{token}", owner="alice") == str((d / "data.csv").resolve())
    with pytest.raises(ValueError) as ei:
        app.resolve_dataset_ref(f"dataset://{token}", owner="bob")
    assert "not found" in str(ei.value).lower()


# ---- bundled refs are globally accessible -----------------------------------

def test_bundled_dataset_not_scoped_by_owner(monkeypatch, tmp_path):
    """Bundled benchmarks ship with the server and are not per-caller."""
    app = _load_app(monkeypatch, tmp_path)
    idx = app.BENCHMARKS_ROOT / "index.json"
    if not idx.exists():
        pytest.skip("no bundled benchmarks index on disk")
    data = json.loads(idx.read_text())
    bundled = data.get("benchmarks") or []
    if not bundled:
        pytest.skip("empty bundled benchmarks index")
    name = bundled[0]["name"]
    for who in (None, "alice", "bob"):
        app.resolve_dataset_ref(f"bundled://{name}", owner=who)


# ---- preset catalogue is served with refs the resolver accepts --------------

def test_presets_only_use_resolvable_model_refs():
    """Every model ref shipped in a preset must be one `resolve_model_ref`
    will accept at request time. Historically a few presets shipped with
    `./checkpoints/<name>` fs paths that only existed on the author's dev
    box - the public server rejected them with 400 the moment the user hit
    "Merge". Pin this never regresses."""
    presets_path = ROOT / "server" / "presets.json"
    data = json.loads(presets_path.read_text())
    bad = []
    for p in data["presets"]:
        body = p.get("body", {})
        refs = list(body.get("models") or [])
        if body.get("base"):
            refs.append(body["base"])
        for ref in refs:
            # Disallow relative or absolute fs paths - the server will not
            # resolve them without MERGESE_ALLOW_LOCAL_PATHS=1, which the
            # public deployment never enables.
            if ref.startswith(("./", "../", "/")) or "\\" in ref:
                bad.append((p["name"], ref))
    assert not bad, f"presets reference non-resolvable paths: {bad}"


# ---- security headers --------------------------------------------------------

def test_security_headers_on_api_response(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "Content-Security-Policy" in r.headers
    assert "challenges.cloudflare.com" in r.headers["Content-Security-Policy"]
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert "frame-ancestors" in r.headers["Content-Security-Policy"]


# ---- library scoping ---------------------------------------------------------

def test_library_scopes_uploads_to_caller(monkeypatch, tmp_path):
    """With auth on, /api/library only lists uploads owned by the caller.
    Also covers the earlier regression where it listed every anonymous
    visitor's uploads to every anonymous visitor."""
    app = _load_app(monkeypatch, tmp_path,
                    MERGESE_REQUIRE_AUTH="1",
                    MERGESE_AUTH_DB=str(tmp_path / "auth.db"),
                    MERGESE_AUTH_SECRET="test-secret")

    store = app._auth_store()
    alice_cid, alice_key = store.mint_key(email="alice@example.com")
    bob_cid, _bob_key = store.mint_key(email="bob@example.com")

    for cid, token in [(alice_cid, "aaaaaaaaaaaa"), (bob_cid, "bbbbbbbbbbbb")]:
        d = app.UPLOADS_ROOT / token
        _make_hf_dir(d)
        app._write_owner(d, cid)

    client = app.app.test_client()
    r = client.get("/api/library", headers={"Authorization": f"Bearer {alice_key}"})
    assert r.status_code == 200
    refs = {u["ref"] for u in r.get_json()["uploads"]}
    assert refs == {"upload://aaaaaaaaaaaa"}
    assert "upload://bbbbbbbbbbbb" not in refs


def test_upload_delete_rejects_cross_tenant(monkeypatch, tmp_path):
    """One caller must not be able to delete another caller's upload by
    guessing / scraping the token."""
    app = _load_app(monkeypatch, tmp_path,
                    MERGESE_REQUIRE_AUTH="1",
                    MERGESE_AUTH_DB=str(tmp_path / "auth.db"),
                    MERGESE_AUTH_SECRET="test-secret")

    store = app._auth_store()
    alice_cid, _alice_key = store.mint_key(email="alice@example.com")
    _bob_cid, bob_key = store.mint_key(email="bob@example.com")

    token = "aaaaaaaaaaaa"
    d = app.UPLOADS_ROOT / token
    _make_hf_dir(d)
    app._write_owner(d, alice_cid)

    client = app.app.test_client()
    r = client.delete(f"/api/uploads/{token}",
                      headers={"Authorization": f"Bearer {bob_key}"})
    # 404 (not 403) so bob can't probe for alice's tokens.
    assert r.status_code == 404
    # Alice's dir is still intact.
    assert d.exists()
