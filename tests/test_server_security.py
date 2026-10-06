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


def test_uncached_hf_ids_in_cmd_dedupes_and_skips_locals(monkeypatch, tmp_path):
    cache = tmp_path / "hf"
    (cache / "hub" / "models--org--cached" / "snapshots" / "s").mkdir(parents=True)
    app = _load_app(monkeypatch, tmp_path, HF_HOME=str(cache))
    local = tmp_path / "a_local_model"; local.mkdir()
    cmd = [
        "merge",
        "org/cached",            # cached -> skip
        "org/uncached-a",        # include
        "org/uncached-b",        # include
        "org/uncached-a",        # duplicate -> skip
        str(local),              # absolute path exists -> skip
        "--flag",                # flag -> skip
    ]
    assert app._uncached_hf_ids_in_cmd(cmd) == ["org/uncached-a", "org/uncached-b"]


def test_prefetch_rejects_oversized_model(monkeypatch, tmp_path):
    """A visitor cannot steer the operator into downloading a 50 GB llama by
    typing its id into the merge form. The size cap stops the fetch before
    any bytes land on disk."""
    app = _load_app(monkeypatch, tmp_path,
                    HF_HOME=str(tmp_path / "hf"),
                    MERGESE_HF_PREFETCH_MAX_BYTES="1048576")  # 1 MB cap

    # Pretend the Hub says the model is 100 MB total across two files.
    class _FakeFile:
        def __init__(self, size): self.size = size
    class _FakeInfo:
        siblings = [_FakeFile(50 * 1024 * 1024), _FakeFile(50 * 1024 * 1024)]
    class _FakeApi:
        def model_info(self, hf_id, files_metadata=False): return _FakeInfo()

    # Stub out the HF Hub module so no network is reached.
    import types
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = lambda: _FakeApi()
    hub.snapshot_download = lambda *a, **kw: pytest.fail(
        "snapshot_download must NOT be called when the size cap is exceeded")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    buf = io.BytesIO()
    with pytest.raises(ValueError) as ei:
        app._prefetch_hf_id("huge/model", buf)
    msg = str(ei.value).lower()
    assert "refusing" in msg and "huge/model" in msg


def test_prefetch_downloads_when_within_cap(monkeypatch, tmp_path):
    app = _load_app(monkeypatch, tmp_path,
                    HF_HOME=str(tmp_path / "hf"),
                    MERGESE_HF_PREFETCH_MAX_BYTES=str(10 * 1024 * 1024))  # 10 MB cap

    class _FakeFile:
        def __init__(self, size): self.size = size
    class _FakeInfo:
        siblings = [_FakeFile(1024)]  # 1 KB total, well under cap
    class _FakeApi:
        def model_info(self, hf_id, files_metadata=False): return _FakeInfo()
    called = {}
    def _dl(hf_id, *a, **kw):
        called["id"] = hf_id
        return "/some/snapshot/path"
    import types
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = lambda: _FakeApi()
    hub.snapshot_download = _dl
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    buf = io.BytesIO()
    local_path = app._prefetch_hf_id("org/tiny", buf)
    # The returned path is what the caller rewrites the worker's cmd to use.
    assert local_path == "/some/snapshot/path"
    assert called["id"] == "org/tiny"
    # Progress is written to the log so visitors can watch it stream.
    log = buf.getvalue().decode()
    assert "looking up org/tiny" in log
    assert "downloading org/tiny" in log
    assert "/some/snapshot/path" in log


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


def test_cached_hf_snapshot_path_resolves_refs_main(monkeypatch, tmp_path):
    """`_cached_hf_snapshot_path` must resolve a cached id by reading refs/main
    and returning the matching snapshot. Needed because loading an HF id
    directly offline is unreliable across transformers versions - we always
    hand the worker the resolved path."""
    cache = tmp_path / "hf"
    model = cache / "hub" / "models--org--mini"
    rev = "deadbeef" * 4
    (model / "snapshots" / rev).mkdir(parents=True)
    (model / "refs").mkdir()
    (model / "refs" / "main").write_text(rev)
    app = _load_app(monkeypatch, tmp_path, HF_HOME=str(cache))
    assert app._cached_hf_snapshot_path("org/mini") == str(model / "snapshots" / rev)
    # Unknown id -> None
    assert app._cached_hf_snapshot_path("org/not-cached") is None


def test_rewrite_hf_ids_to_local_paths(monkeypatch, tmp_path):
    cache = tmp_path / "hf"
    model = cache / "hub" / "models--org--cached"
    rev = "abc1234567890abc"
    (model / "snapshots" / rev).mkdir(parents=True)
    (model / "refs").mkdir()
    (model / "refs" / "main").write_text(rev)
    app = _load_app(monkeypatch, tmp_path, HF_HOME=str(cache))
    local_abs = str(tmp_path / "already_a_path"); Path(local_abs).mkdir()
    cmd = [
        "merge",
        "org/cached",         # -> rewrite to local snapshot
        "org/not-cached",     # -> leave as id (caller will refuse later)
        local_abs,            # -> leave: absolute path
        "--trim-percentile",  # -> leave: flag
        "20",                 # -> leave: numeric literal, no slash
    ]
    out = app._rewrite_hf_ids_to_local_paths(cmd)
    assert out == [
        "merge",
        str(model / "snapshots" / rev),
        "org/not-cached",
        local_abs,
        "--trim-percentile",
        "20",
    ]
