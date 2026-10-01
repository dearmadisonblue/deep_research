from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from . import adapters
from .catalog import Asset, sha256_text
from .registry import BASE_MODEL_ID, BASE_REVISION, MODEL_REGISTRY
from .runtime import _STATE, _runtime
from .workflows import ImageUse, _DrawSpec


def set_logging(enabled: bool) -> None:
    """Enable or disable generation provenance logging."""
    if not isinstance(enabled, bool):
        raise TypeError("set_logging() requires a boolean")
    _STATE.logging_enabled = enabled


def is_logging() -> bool:
    """Return whether generation provenance logging is enabled."""
    return _STATE.logging_enabled


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def atomic_save_png(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)
    return path


def _relative_to_root(path: Path) -> str:
    try:
        return path.relative_to(_runtime().paths.root_dir).as_posix()
    except ValueError:
        return str(path)


def _reference_record(image_use: ImageUse, index: int, model_id: str) -> dict[str, Any]:
    source = image_use.source
    if isinstance(source, Asset):
        record: dict[str, Any] = {
            "id": source.id,
            "kind": source.kind,
            "path": _relative_to_root(source.path),
            "sha256": source.content_sha256(),
        }
    else:
        digest = hashlib.sha256()
        digest.update(f"{source.mode}:{source.width}x{source.height}:".encode())
        digest.update(source.tobytes())
        record = {
            "id": None,
            "kind": "image",
            "path": None,
            "sha256": digest.hexdigest(),
        }
    return {
        **record,
        "role": "reference",
        "qwen_token": f"<image{index}>" if model_id == "qwen" else None,
        "strength": image_use.strength,
    }


def _build_log_payload(
    *,
    spec: _DrawSpec,
    run_id: str,
    timestamp_utc: str,
    timestamp_nyc: str,
    seeds: list[int],
    images: list[Image.Image],
    output_paths: list[Path | None],
    memory_before: dict[str, Any],
    memory_after: dict[str, Any],
) -> dict[str, Any]:
    prompt_obj = spec.prompt_obj
    model_cfg = MODEL_REGISTRY[spec.model_id]
    components = {
        name: {
            "filename": component["filename"],
            "repo_id": component["repo_id"],
            "repo_filename": component["repo_filename"],
            "revision": component["revision"],
            "size": component["size"],
            "sha256": component["sha256"],
        }
        for name, component in model_cfg["components"].items()
    }
    return {
        "schema": 2,
        "type": "generation_run",
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "timestamp_nyc": timestamp_nyc,
        "workflow": {
            "name": spec.workflow_name,
            "source_sha256": spec.workflow_source_sha256,
        },
        "prompt": {
            "id": prompt_obj.id if prompt_obj is not None else None,
            "path": _relative_to_root(prompt_obj.path)
            if prompt_obj is not None
            else None,
            "source_sha256": (
                prompt_obj.source_sha256
                if prompt_obj is not None
                else sha256_text(spec.prompt)
            ),
            "text": spec.prompt,
            "metadata": dict(prompt_obj.metadata) if prompt_obj is not None else {},
        },
        "generation": {
            "seeds": seeds,
            "width": spec.width,
            "height": spec.height,
            "size_explicit": spec.size_explicit,
            "steps": spec.steps,
            "guidance": spec.guidance,
            "sampler": spec.sampler,
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "negative_prompt": spec.negative_prompt,
            "batch": spec.batch,
            "saving_enabled": spec.saving_enabled,
        },
        "assets": [
            _reference_record(image_use, index, spec.model_id)
            for index, image_use in enumerate(spec.images, start=1)
        ],
        "adapters": [
            {
                "id": item.id,
                "strength": item.strength,
                "path": _relative_to_root(adapters._get_adapter(item.id).path),
                "sha256": adapters._adapter_sha256(item.id),
                "model_id": adapters._get_adapter(item.id).model_id,
            }
            for item in spec.adapters
        ],
        "model": {
            "id": spec.model_id,
            "display_name": model_cfg["display_name"],
            "quantization": model_cfg["quantization"],
            "license": model_cfg["license"],
            "components": components,
            "pipeline_config": {"repo_id": BASE_MODEL_ID, "revision": BASE_REVISION},
            "backend": "diffusers",
            "backend_versions": _runtime().backend_versions,
            "model_cpu_offload": bool(
                getattr(_STATE.model["pipeline"], "_all_hooks", [])
            ),
        },
        "runtime_memory": {
            "before_generation": memory_before,
            "after_generation": memory_after,
            "latest_model_switch": _STATE.model_switch_history[-1]
            if _STATE.model_switch_history
            else None,
        },
        "outputs": [
            {
                "path": _relative_to_root(path) if path is not None else None,
                "seed": seed,
                "mode": image.mode,
                "has_alpha": image.mode == "RGBA",
            }
            for path, seed, image in zip(output_paths, seeds, images)
        ],
    }
