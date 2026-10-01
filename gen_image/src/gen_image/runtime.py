from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ModelId, SecretProvider


@dataclass(frozen=True)
class RuntimePaths:
    root_dir: Path
    temp_dir: Path
    prompts_dir: Path
    assets_dir: Path
    photos_dir: Path
    videos_dir: Path
    logs_dir: Path
    run_logs_dir: Path
    models_dir: Path
    hf_files_dir: Path
    adapters_dir: Path
    local_hf_dir: Path
    local_hf_hub_dir: Path
    local_hf_xet_dir: Path


@dataclass
class _Runtime:
    paths: RuntimePaths
    secret_provider: SecretProvider | None
    torch: Any
    backend_versions: dict[str, str]


_RUNTIME: _Runtime | None = None


def _build_paths(root_dir: str | Path, temp_dir: str | Path) -> RuntimePaths:
    root = Path(root_dir).expanduser().resolve()
    temp = Path(temp_dir).expanduser().resolve()
    models = root / "models"
    local_hf = temp / "huggingface"
    return RuntimePaths(
        root_dir=root,
        temp_dir=temp,
        prompts_dir=root / "prompts",
        assets_dir=root / "assets",
        photos_dir=root / "media" / "photos",
        videos_dir=root / "media" / "videos",
        logs_dir=root / "logs",
        run_logs_dir=root / "logs" / "runs",
        models_dir=models,
        hf_files_dir=models / "huggingface" / "files",
        adapters_dir=models / "adapters",
        local_hf_dir=local_hf,
        local_hf_hub_dir=local_hf / "hub",
        local_hf_xet_dir=local_hf / "xet",
    )


def _runtime() -> _Runtime:
    if _RUNTIME is None:
        raise RuntimeError(
            "gen_image is not initialized. Call await gen_image.init(...)."
        )
    return _RUNTIME


@dataclass
class _SessionState:
    model: dict[str, Any] = field(default_factory=dict)
    model_switch_history: list[dict[str, Any]] = field(default_factory=list)
    model_file_paths: dict[ModelId, dict[str, Path]] = field(default_factory=dict)
    logging_enabled: bool = True
    saving_enabled: bool = True
    random_seed: int | None = None


_STATE = _SessionState()


def set_saving(enabled: bool) -> None:
    """Enable or disable saving generated images to files."""
    if not isinstance(enabled, bool):
        raise TypeError("set_saving() requires a boolean")
    _STATE.saving_enabled = enabled


def is_saving() -> bool:
    """Return whether generated images are saved to files."""
    return _STATE.saving_enabled


def set_random_seed(seed: int | None) -> None:
    """Set a fixed generation seed, or restore automatic seeding with None."""
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("set_random_seed() requires an integer or None")
        if not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must be between 0 and 2**64 - 1")
    _STATE.random_seed = seed


def _runtime_or_none() -> _Runtime | None:
    return _RUNTIME


async def init(
    root_dir: str | Path,
    temp_dir: str | Path,
    *,
    secret_provider: SecretProvider | None = None,
) -> Mapping[str, Any]:
    """Initialize paths, caches, Diffusers dependencies, and file indexes."""
    global _RUNTIME

    paths = _build_paths(root_dir, temp_dir)
    if _RUNTIME is not None:
        if (
            _RUNTIME.paths.root_dir == paths.root_dir
            and _RUNTIME.paths.temp_dir == paths.temp_dir
        ):
            from .diagnostics import status

            return status()
        raise RuntimeError(
            "gen_image is already initialized with different directories."
        )

    for path in (
        paths.root_dir,
        paths.temp_dir,
        paths.prompts_dir,
        paths.assets_dir,
        paths.photos_dir,
        paths.videos_dir,
        paths.run_logs_dir,
        paths.hf_files_dir,
        paths.adapters_dir / "qwen",
        paths.local_hf_hub_dir,
        paths.local_hf_xet_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(paths.local_hf_dir)
    os.environ["HF_HUB_CACHE"] = str(paths.local_hf_hub_dir)
    os.environ["HF_XET_CACHE"] = str(paths.local_hf_xet_dir)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"

    from .diffusers_backend import initialize_backend

    backend = await initialize_backend(paths)
    _RUNTIME = _Runtime(paths=paths, secret_provider=secret_provider, **backend)

    from .adapters import refresh_adapters
    from .catalog import refresh_library
    from .diagnostics import status

    refresh_library()
    refresh_adapters()
    print("Diffusers backend:", _RUNTIME.backend_versions)
    print("gen_image initialized:", paths.root_dir)
    return status()
