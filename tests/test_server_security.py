"""Security-hardening tests for the web server.

These exercise the upload guards (pickle rejection, zip-bomb defenses) and the
offline-worker plumbing (HF cache detection, sandbox command construction)
without needing a running Flask server or any real checkpoints.
"""
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest

# The server imports Flask at module load; skip this whole file gracefully if
# the server extra isn't installed rather than erroring at collection.
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


def _load_app(monkeypatch, tmp_path, **env):
    """Import server/app.py fresh with an isolated uploads/artifacts root."""
    monkeypatch.setenv("MERGESE_UPLOADS", str(tmp_path / "uploads"))
    monkeypatch.setenv("MERGESE_ARTIFACTS", str(tmp_path / "artifacts"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("mergese_app", ROOT / "server" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mergese_app"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- pickle / unsafe-file rejection -----------------------------------------

def test_scan_rejects_pickle_bin(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "pytorch_model.bin").write_bytes(b"\x80\x04junk")
    err = app._scan_unsafe_files(d)
    assert err and "pickle" in err.lower()


def test_scan_rejects_code_files(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "evil.py").write_text("import os; os.system('id')")
    err = app._scan_unsafe_files(d)
    assert err is not None and "code" in err.lower()


def test_scan_allows_safetensors(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00" * 16)
    assert app._scan_unsafe_files(d) is None


def test_validate_requires_safetensors(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "pytorch_model.bin").write_bytes(b"\x80\x04junk")
    # Even though a weights file exists, a pickle .bin must be refused.
    err = app._validate_hf_dir(d)
    assert err is not None


def test_pickle_allowed_when_opted_in(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path, MERGESE_ALLOW_PICKLE_UPLOADS="1")
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "pytorch_model.bin").write_bytes(b"\x80\x04junk")
    assert app._scan_unsafe_files(d) is None
    assert app._validate_hf_dir(d) is None


# ---- zip-bomb defenses -------------------------------------------------------

def _zip_with(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members:
            z.writestr(name, data)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_zip_extract_blocks_traversal(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    zf = _zip_with([("../escape.txt", b"x")])
    err, _ = app._safe_zip_extract(zf, tmp_path / "dest")
    assert err and "escape" in err.lower()


def test_zip_extract_blocks_too_many_entries(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path, MERGESE_MAX_ARCHIVE_ENTRIES="5")
    zf = _zip_with([(f"f{i}.txt", b"x") for i in range(10)])
    dest = tmp_path / "dest"
    dest.mkdir()
    err, _ = app._safe_zip_extract(zf, dest)
    assert err and "many entries" in err.lower()


def test_zip_extract_blocks_uncompressed_bomb(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path, MERGESE_MAX_UNCOMPRESSED_BYTES="1024")
    zf = _zip_with([("big.txt", b"A" * 8192)])
    dest = tmp_path / "dest"
    dest.mkdir()
    err, _ = app._safe_zip_extract(zf, dest)
    assert err and "limit" in err.lower()


def test_zip_extract_accepts_normal_archive(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    zf = _zip_with([("config.json", b"{}"), ("model.safetensors", b"\x00" * 32)])
    dest = tmp_path / "dest"
    dest.mkdir()
    err, extracted = app._safe_zip_extract(zf, dest)
    assert err is None
    assert set(extracted) == {"config.json", "model.safetensors"}
    assert (dest / "model.safetensors").exists()


# ---- offline-worker plumbing -------------------------------------------------

def test_hf_cache_detection(monkeypatch, tmp_path):
    cache = tmp_path / "hf"
    snap = cache / "hub" / "models--org--model" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    app = _load_app(monkeypatch, tmp_path, HF_HOME=str(cache))
    assert app._hf_id_is_cached("org/model") is True
    assert app._hf_id_is_cached("org/not-there") is False


def test_cmd_needs_network_flags_uncached_id(monkeypatch, tmp_path):
    cache = tmp_path / "hf"
    (cache / "hub" / "models--org--cached" / "snapshots" / "s").mkdir(parents=True)
    app = _load_app(monkeypatch, tmp_path, HF_HOME=str(cache))
    # A cached id needs no network; an uncached one does.
    assert app._cmd_needs_network(["merge", "org/cached", "--base", "org/cached"]) is False
    assert app._cmd_needs_network(["merge", "org/uncached"]) is True
    # Absolute local paths never count as network.
    assert app._cmd_needs_network(["merge", str(tmp_path), "--flag"]) is False


def test_build_worker_uses_stripped_env_and_job_dir(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path)
    # A secret in the server env must NOT reach the worker.
    monkeypatch.setenv("MERGESE_DB_PASSWORD", "supersecret")
    job = app.Job(id="testjob", cmd=["/bin/echo", "hi"], kind="merge")
    (app.ARTIFACTS_ROOT / "testjob").mkdir(parents=True, exist_ok=True)

    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kw):
            captured["cmd"] = cmd
            captured["env"] = kw.get("env")
            captured["cwd"] = kw.get("cwd")
            self.pid = 1234

    monkeypatch.setattr(app.subprocess, "Popen", _FakePopen)
    devnull = io.BytesIO()
    app._build_worker(job, devnull)

    assert "MERGESE_DB_PASSWORD" not in captured["env"]
    assert captured["env"]["HOME"].endswith("/testjob/home")
    assert captured["cwd"].endswith("/testjob")
    # When the host supports a netns, the command is wrapped with unshare.
    if app.NETNS_OK:
        assert captured["cmd"][0] == app._UNSHARE_BIN
        assert "-rn" in captured["cmd"]
