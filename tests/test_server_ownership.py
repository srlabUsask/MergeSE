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


# ---- B03: hf:// prefix must not accept absolute-path smuggling --------------

def test_hf_prefix_rejects_absolute_path(monkeypatch, tmp_path):
    """`hf:///tmp/evil` previously returned `/tmp/evil` verbatim, bypassing
    ALLOW_LOCAL_PATHS=0. The resolver must verify the remainder is a valid
    HuggingFace org/name id."""
    app = _load_app(monkeypatch, tmp_path)  # ALLOW_LOCAL_PATHS default off
    for bad in ("hf:///tmp/evil", "hf:////etc/passwd", "hf://../../etc"):
        with pytest.raises(ValueError, match="HuggingFace Hub id"):
            app.resolve_model_ref(bad)
    # Well-formed hf:// still works.
    assert app.resolve_model_ref("hf://microsoft/codebert-base") == "microsoft/codebert-base"


# ---- B04: path containment must be component-aware --------------------------

def test_server_prefix_rejects_sibling_dir(monkeypatch, tmp_path):
    """`server://../checkpoints_private` previously resolved outside the
    allowed root because str.startswith on `/root/checkpoints` matched
    `/root/checkpoints_private`. Switch to Path.relative_to catches it."""
    allowed = tmp_path / "checkpoints"; allowed.mkdir()
    sibling = tmp_path / "checkpoints_private"; sibling.mkdir()
    (sibling / "config.json").write_text("{}")
    app = _load_app(monkeypatch, tmp_path, MERGESE_CHECKPOINTS=str(allowed))
    with pytest.raises(ValueError, match="escapes checkpoints root"):
        app.resolve_model_ref("server://../checkpoints_private")


def test_server_dataset_prefix_rejects_sibling_dir(monkeypatch, tmp_path):
    allowed = tmp_path / "datasets"; allowed.mkdir()
    sibling = tmp_path / "datasets_private"; sibling.mkdir()
    (sibling / "secret.csv").write_text("code,label\nx,1")
    app = _load_app(monkeypatch, tmp_path, MERGESE_DATASETS=str(allowed))
    with pytest.raises(ValueError, match="escapes datasets root"):
        app.resolve_dataset_ref("server-dataset://../datasets_private/secret.csv")


# ---- B14: malformed JSON body (list, string, number) must 400 ---------------

def test_json_body_rejects_non_object(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    for payload in (["bad"], "bad", 42, True, None):
        import json as _json
        body = _json.dumps(payload) if payload is not None else "null"
        for ep in ("/api/inspect", "/api/merge", "/api/evaluate", "/api/export"):
            r = client.post(ep, data=body, content_type="application/json")
            # Expect 400 with a JSON error - never 500 and never HTML.
            assert r.status_code in (400,), f"{ep} with {body!r} gave {r.status_code}"
            assert r.content_type.startswith("application/json"), f"{ep} not JSON"


# ---- B15: cancel of a pending job must mark it cancelled --------------------

def test_cancel_pending_job(monkeypatch, tmp_path):
    """Previously `cancel` on a pending job returned {ok:false,status:pending}
    because no process existed yet. The new handler marks the job cancelled
    and _run_job checks that flag before launching the worker."""
    app = _load_app(monkeypatch, tmp_path)
    # Hand-build a pending Job in JOBS and attempt to cancel it.
    jid = "abc123def456"
    (app.ARTIFACTS_ROOT / jid).mkdir(parents=True, exist_ok=True)
    job = app.Job(
        id=jid, cmd=["/bin/sleep", "60"], kind="merge",
        log_path=app.ARTIFACTS_ROOT / jid / "log.txt",
    )
    with app.JOBS_LOCK:
        app.JOBS[jid] = job
    client = app.app.test_client()
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["status"] == "cancelled"
    assert app.JOBS[jid].status == "cancelled"
    assert app.JOBS[jid].finished_at is not None


# ---- B03 (retest) — hf:// must reject dot-component path smuggling ----------

def test_hf_prefix_rejects_relative_path_components(monkeypatch, tmp_path):
    """Regex that let `.` / `..` through as namespace components allowed
    hf://../secret -> ../secret to leak out of the resolver. Tighten to a
    strict HF-component regex that requires each component start with an
    alphanumeric / underscore / hyphen (not a dot)."""
    app = _load_app(monkeypatch, tmp_path)
    for bad in ("hf://../secret", "hf://./secret", "hf://.", "hf://..",
                "hf://../../etc/passwd", "hf://foo/./bar", "hf://foo/../bar"):
        with pytest.raises(ValueError, match="HuggingFace Hub id"):
            app.resolve_model_ref(bad)
    # Legit ids with dots inside a component are still fine (model-y.1 etc).
    assert app.resolve_model_ref("hf://microsoft/codebert-base") == "microsoft/codebert-base"
    assert app.resolve_model_ref("hf://org_x/model-y.1") == "org_x/model-y.1"


# ---- B14 (retest) — invalid field types inside a dict body must 400 --------

def test_json_body_rejects_bad_field_types(monkeypatch, tmp_path):
    """`{"models": 42}` previously 500'd because the handler did `len(42)` on
    an int. Per-field validators now return structured 400 responses."""
    app = _load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    bad_payloads = {
        "/api/inspect": [
            {"models": 42},
            {"models": [1, 2]},             # list of non-strings
            {"models": ["a", "b"], "base": 7},
        ],
        "/api/merge": [
            {"models": 42, "base": "org/x"},
            {"models": ["a", "b"], "base": "org/x", "method": "nonsense"},
            {"models": ["a", "b"], "base": "org/x", "trim_percentile": "high"},
        ],
        "/api/evaluate": [
            {"model": 42},
            {"model": "org/x", "batch_size": "one"},
            {"model": "org/x", "metric": "oops"},
        ],
        "/api/export": [
            {"model": 42},
            {"model": "org/x", "format": "parquet"},  # unknown enum
        ],
    }
    for ep, payloads in bad_payloads.items():
        for body in payloads:
            r = client.post(ep, json=body, content_type="application/json")
            assert r.status_code == 400, f"{ep} with {body!r} returned {r.status_code}"
            assert r.content_type.startswith("application/json")
            msg = r.get_json()["error"]
            # Message should mention the bad field, not just "bad request".
            assert any(k in msg for k in body.keys()) or "format" in msg or "metric" in msg


# ---- B14 (retest round 2) — dataset_ref type + NaN numerics -----------------

def test_evaluate_rejects_non_string_dataset_ref(monkeypatch, tmp_path):
    """`{"model": "org/x", "dataset_ref": 42}` previously 500'd because
    api_evaluate read dataset_ref via raw body.get and passed an int into
    resolve_dataset_ref's string ops. Validate as a string first."""
    app = _load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    for bad in (42, ["bundled://bigclonebench"], {"ref": "x"}, 3.14):
        r = client.post("/api/evaluate",
                        json={"model": "org/x", "dataset_ref": bad},
                        content_type="application/json")
        assert r.status_code == 400, f"dataset_ref={bad!r} -> {r.status_code}"
        assert "dataset_ref" in r.get_json()["error"]


def test_number_fields_reject_nan_and_inf(monkeypatch, tmp_path):
    """`batch_size: NaN` previously reached `int(NaN)` and 500'd. NaN/Inf
    pass the type check AND skate past range comparisons (all comparisons
    with NaN are False), so the finiteness check has to be explicit."""
    app = _load_app(monkeypatch, tmp_path)
    client = app.app.test_client()
    import math as _m
    for bad in (_m.nan, _m.inf, -_m.inf):
        r = client.post("/api/evaluate",
                        json={"model": "org/x", "batch_size": bad},
                        content_type="application/json")
        assert r.status_code == 400, f"batch_size={bad!r} -> {r.status_code}"
        err = r.get_json()["error"].lower()
        assert "batch_size" in err and ("finite" in err or "nan" in err.lower() or "inf" in err.lower())
    # Fractional values for int fields are also rejected (3.14 -> int(3.14)=3
    # would silently truncate).
    r = client.post("/api/evaluate",
                    json={"model": "org/x", "batch_size": 3.14},
                    content_type="application/json")
    assert r.status_code == 400
    assert "integer" in r.get_json()["error"].lower()


# ---- concurrency hardening (external pressure-test findings) ---------------

def test_atomic_admission_respects_queue_cap(monkeypatch, tmp_path):
    """P1-C: _capacity_response + _reserve + _new_job wasn't atomic, so N
    concurrent requests all passed the capacity check before any inserted.
    `_admit_job` folds those into one critical section; the queue cap must
    now hold under racing admissions."""
    import threading
    app = _load_app(monkeypatch, tmp_path,
                    MERGESE_MAX_CONCURRENT="2",
                    MERGESE_MAX_QUEUE="4")
    admitted, rejected = [], []
    errors = []

    def _one():
        try:
            jid, _ = app._admit_job(owner=None, kind="pending")
            admitted.append(jid)
        except app._CapacityError:
            rejected.append(1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_one) for _ in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, f"unexpected exceptions: {errors}"
    assert len(admitted) == 4, f"queue cap=4; got {len(admitted)} admitted"
    assert len(admitted) + len(rejected) == 20


def test_atomic_admission_respects_per_user_cap(monkeypatch, tmp_path):
    """P2: per-user active-slot limit raced the JOBS insert, so one key could
    overshoot its tier.max_active (observed 7/3 under pressure). Atomic
    admission must hold the cap per owner."""
    import threading
    app = _load_app(monkeypatch, tmp_path,
                    MERGESE_REQUIRE_AUTH="1",
                    MERGESE_AUTH_DB=str(tmp_path / "auth.db"),
                    MERGESE_AUTH_SECRET="test-secret",
                    MERGESE_ANON_MAX_ACTIVE="3",
                    MERGESE_KEY_MAX_ACTIVE="3",
                    MERGESE_KEY_DAILY_JOBS="1000",
                    MERGESE_MAX_QUEUE="100")

    store = app._auth_store()
    _cid, key = store.mint_key(email="alice@local")

    admitted, rejected = [], []
    errors = []

    def _one():
        with app.app.test_request_context(
                headers={"Authorization": f"Bearer {key}"}):
            try:
                jid, _ = app._admit_job(owner=None, kind="pending")
                admitted.append(jid)
            except app._CapacityError:
                rejected.append(1)
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=_one) for _ in range(16)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, f"unexpected exceptions: {errors}"
    assert len(admitted) == 3, (
        f"KEY_MAX_ACTIVE=3; one key got {len(admitted)} slots")


def test_auth_store_sqlite_access_is_serialised(monkeypatch, tmp_path):
    """P1-A: concurrent reads on the shared sqlite3 connection surfaced as
    intermittent 401 + sqlite3.InterfaceError. Hammer _client_from_key from
    many threads with a known-good key; every call must succeed."""
    import threading
    app = _load_app(monkeypatch, tmp_path,
                    MERGESE_REQUIRE_AUTH="1",
                    MERGESE_AUTH_DB=str(tmp_path / "auth.db"),
                    MERGESE_AUTH_SECRET="test-secret")
    import auth as _authmod
    store = app._auth_store()
    _cid, key = store.mint_key(email="alice@local")

    bad, errs = [], []
    def _hit():
        try:
            c = store.authenticate(api_key=key, anon_token=None)
            if c.client_id != _cid: bad.append(c.client_id)
        except _authmod.AuthError:
            bad.append("autherror")
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=_hit) for _ in range(64)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errs, f"sqlite3 raised under concurrency: {errs}"
    assert not bad, f"{len(bad)} valid-key calls failed under concurrency"


def test_download_zip_build_is_atomic(monkeypatch, tmp_path):
    """P1-B: the download handler built _download.zip at the final path, so
    concurrent requests could stream a half-written file. Build to .inflight
    + os.replace means every reader sees either absent or complete."""
    import threading, zipfile
    app = _load_app(monkeypatch, tmp_path)
    # Fake a merge job with a 'merged' dir containing a few files.
    jid = "concur123abc"
    jroot = app.ARTIFACTS_ROOT / jid
    merged = jroot / "merged"
    merged.mkdir(parents=True)
    for i in range(4):
        (merged / f"part{i}.bin").write_bytes(b"X" * 1024)
    (merged / "config.json").write_text("{}")

    # Simulate a merge job in JOBS so the handler finds the artifact.
    job = app.Job(id=jid, cmd=["/bin/true"], kind="merge",
                  log_path=jroot / "log.txt")
    job.status = "done"
    with app.JOBS_LOCK:
        app.JOBS[jid] = job

    client = app.app.test_client()
    sizes = []
    errs = []

    def _grab():
        try:
            r = client.get(f"/api/jobs/{jid}/download")
            # Load entire body (test client allows this safely).
            data = r.get_data()
            sizes.append(len(data))
            with zipfile.ZipFile(__import__("io").BytesIO(data)) as zf:
                assert len(zf.namelist()) == 5, zf.namelist()
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=_grab) for _ in range(16)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errs, f"download thread errors: {errs}"
    # Every zip opened cleanly with 5 entries; by that point the sizes should
    # all match too (one atomic build, 16 reads of the same file).
    assert len(set(sizes)) == 1, f"zip sizes diverge: {set(sizes)}"
