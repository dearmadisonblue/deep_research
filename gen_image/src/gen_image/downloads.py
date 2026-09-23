from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from .registry import MODEL_REGISTRY, normalize_model_id
from .runtime import _STATE, _runtime


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_safetensors(path: Path, expected_size: int) -> tuple[bool, str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(160)
    except FileNotFoundError:
        return False, "missing"
    except Exception as exc:
        return False, f"cannot read file: {exc}"
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        return False, "Git-LFS pointer instead of model bytes"
    if prefix.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        return False, "HTML response instead of model bytes"
    if size != expected_size:
        return False, f"wrong size: {size:,}; expected {expected_size:,}"
    try:
        with path.open("rb") as stream:
            raw_len = stream.read(8)
            if len(raw_len) != 8:
                return False, "missing safetensors header length"
            header_len = int.from_bytes(raw_len, "little", signed=False)
            if header_len <= 1 or header_len > 256 * 1024 * 1024:
                return False, f"implausible safetensors header length: {header_len:,}"
            header = stream.read(header_len)
            if len(header) != header_len:
                return False, "truncated safetensors header"
            json.loads(header)
    except Exception as exc:
        return False, f"invalid safetensors header: {exc}"
    return True, "ok"


def _verified_marker(path: Path) -> Path:
    return path.with_name(path.name + ".verified.json")


def _write_verified_marker(
    path: Path,
    *,
    repo_id: str,
    filename: str,
    expected_size: int,
    expected_sha256: str,
) -> None:
    _verified_marker(path).write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "filename": filename,
                "size": expected_size,
                "sha256": expected_sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _marker_matches(
    path: Path,
    *,
    repo_id: str,
    filename: str,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    marker = _verified_marker(path)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload == {
        "repo_id": repo_id,
        "filename": filename,
        "size": expected_size,
        "sha256": expected_sha256,
    }


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip()
    provider = _runtime().secret_provider
    if provider is None:
        return None
    value = provider(name)
    return value.strip() if value else None


def _hf_error_requires_auth(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return isinstance(exc, (GatedRepoError, RepositoryNotFoundError)) or getattr(
        response, "status_code", None
    ) in (401, 403)


def _hf_download(repo_id: str, filename: str) -> Path:
    rt = _runtime()
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(rt.paths.local_hf_hub_dir),
                force_download=True,
                token=False,
            )
        )
    except Exception as anonymous_error:
        if not _hf_error_requires_auth(anonymous_error):
            raise
        token = _secret("HF_TOKEN")
        if not token:
            raise RuntimeError(
                f"Authentication is required for {repo_id} :: {filename}. "
                "Provide HF_TOKEN through the environment or secret_provider."
            ) from anonymous_error
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(rt.paths.local_hf_hub_dir),
                force_download=True,
                token=token,
            )
        )


def _persistent_download(
    *, repo_id: str, filename: str, expected_size: int, expected_sha256: str
) -> Path:
    rt = _runtime()
    target = rt.paths.hf_files_dir / repo_id.replace("/", "--") / Path(filename)
    if target.exists():
        ok, reason = _looks_like_safetensors(target, expected_size)
        if ok and _marker_matches(
            target,
            repo_id=repo_id,
            filename=filename,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        ):
            print("Persistent cache hit:", repo_id, "::", filename)
            return target
        if ok and _sha256(target).lower() == expected_sha256.lower():
            _write_verified_marker(
                target,
                repo_id=repo_id,
                filename=filename,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            return target
        print("Discarding invalid persistent file:", target, "(", reason, ")")
        target.unlink(missing_ok=True)
        _verified_marker(target).unlink(missing_ok=True)

    downloaded = _hf_download(repo_id, filename)
    ok, reason = _looks_like_safetensors(downloaded, expected_size)
    if not ok:
        raise RuntimeError(f"Downloaded file is invalid: {reason}")
    actual = _sha256(downloaded)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {repo_id} :: {filename}: "
            f"expected {expected_sha256}, got {actual}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(downloaded, temporary)
    temporary.replace(target)
    _write_verified_marker(
        target,
        repo_id=repo_id,
        filename=filename,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    return target


def _component_target_dir(category: str) -> Path:
    paths = _runtime().paths
    return {
        "diffusion_models": paths.diffusion_models_dir,
        "text_encoders": paths.text_encoders_dir,
        "vae": paths.vae_dir,
    }[category]


def _expose_to_comfy(cached_path: Path, target_dir: Path, filename: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.is_symlink() or target.exists():
        try:
            if target.resolve() == cached_path.resolve():
                return target
        except Exception:
            pass
        target.unlink()
    try:
        target.symlink_to(cached_path)
    except OSError:
        shutil.copy2(cached_path, target)
    return target


def _ensure_model_files(name: str) -> dict[str, Path]:
    model_id = normalize_model_id(name)
    existing = _STATE.model_file_paths.get(model_id)
    if existing and all(path.exists() for path in existing.values()):
        return existing
    paths: dict[str, Path] = {}
    for component_name, component in MODEL_REGISTRY[model_id]["components"].items():
        cached = _persistent_download(
            repo_id=component["repo_id"],
            filename=component["repo_filename"],
            expected_size=component["size"],
            expected_sha256=component["sha256"],
        )
        paths[component_name] = _expose_to_comfy(
            cached, _component_target_dir(component["category"]), component["filename"]
        )
    _STATE.model_file_paths[model_id] = paths
    return paths
