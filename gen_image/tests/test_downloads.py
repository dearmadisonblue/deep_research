from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
from gen_image.registry import BASE_MODEL_ID, BASE_REVISION

from gen_image import downloads, models, runtime


def test_pinned_download_cache_rechecks_modified_files(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b'{"config":1}')
    calls = []
    monkeypatch.setattr(
        downloads,
        "_runtime",
        lambda: SimpleNamespace(
            paths=SimpleNamespace(hf_files_dir=tmp_path / "persistent")
        ),
    )

    def download(repo, name, revision):
        calls.append((repo, name, revision))
        return source

    monkeypatch.setattr(downloads, "_hf_download", download)
    kwargs = dict(
        repo_id="Qwen/test",
        revision="pinned",
        filename="transformer/config.json",
        expected_size=source.stat().st_size,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    target = downloads._persistent_download(**kwargs)
    assert downloads._persistent_download(**kwargs) == target
    assert calls == [("Qwen/test", "transformer/config.json", "pinned")]
    target.write_bytes(b'{"config":2}')  # Same size, different bytes.
    assert downloads._persistent_download(**kwargs).read_bytes() == source.read_bytes()
    assert len(calls) == 2


def test_downloads_only_use_published_components_and_config_allowlist(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        downloads,
        "_runtime",
        lambda: SimpleNamespace(paths=SimpleNamespace(hf_files_dir=tmp_path)),
    )

    def download(**kwargs):
        calls.append(kwargs)
        return tmp_path / kwargs["filename"]

    monkeypatch.setattr(downloads, "_persistent_download", download)
    paths = downloads._ensure_model_files("qwen")
    assert set(paths) == {"diffusion", "text_encoder", "vae", "pipeline_config"}
    assert [call["repo_id"] for call in calls[:3]] == ["unsloth/Qwen-Image-2.1-FP8"] * 3
    for call in calls[3:]:
        assert (call["repo_id"], call["revision"]) == (BASE_MODEL_ID, BASE_REVISION)
        assert not call["filename"].endswith(
            (".safetensors", ".safetensors.index.json")
        )
    assert all(
        len(call["expected_sha256"]) == 64 and call["revision"] for call in calls
    )


def test_no_model_download_on_unsupported_hardware(monkeypatch):
    monkeypatch.setattr(runtime._STATE, "model", {})
    fake_runtime = SimpleNamespace(
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setattr(models, "_runtime", lambda: fake_runtime)

    def download(_):
        pytest.fail("should reject CPU before downloading models")

    monkeypatch.setattr(models, "_ensure_model_files", download)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        models.load_model("qwen")
