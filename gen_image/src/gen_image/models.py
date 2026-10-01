from __future__ import annotations

import gc
from datetime import UTC, datetime
from typing import Any

from . import adapters
from .config import ModelId
from .diffusers_backend import load_pipeline
from .downloads import _ensure_model_files
from .registry import MODEL_REGISTRY, normalize_model_id
from .runtime import _STATE, _runtime


def current_model() -> ModelId | None:
    """Return the active model identifier, or None when none is loaded."""
    return _STATE.model.get("model_id")


def _release_active_model() -> None:
    rt = _runtime()
    adapters.clear_adapter_cache(files=False, models=True)
    adapters._clear_default_adapters()
    _STATE.model.clear()
    gc.collect()
    if rt.torch.cuda.is_available():
        rt.torch.cuda.empty_cache()
        rt.torch.cuda.synchronize()


def _load_model_stack(model_id: ModelId) -> dict[str, Any]:
    spec = MODEL_REGISTRY[model_id]
    paths = _ensure_model_files(model_id)
    pipeline = load_pipeline(paths)
    return {
        "model_id": model_id,
        "spec": spec,
        "paths": paths,
        "pipeline": pipeline,
        "loaded_at_utc": datetime.now(UTC).isoformat(),
    }


def load_model(name: str, *, force_reload: bool = False) -> ModelId:
    """Download missing files, verify them, and activate a model stack."""
    from .diagnostics import print_runtime_memory

    _runtime()
    model_id = normalize_model_id(name)
    previous_id = current_model()
    if previous_id == model_id and not force_reload:
        return model_id
    rt = _runtime()
    if not rt.torch.cuda.is_available() or not rt.torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen 2.1 INT8/FP8 requires a CUDA GPU with BF16 support.")
    _ensure_model_files(model_id)
    before = print_runtime_memory(
        f"before model switch ({previous_id or 'none'} -> {model_id})"
    )
    rt = _runtime()
    if rt.torch.cuda.is_available():
        rt.torch.cuda.reset_peak_memory_stats()
    _release_active_model()
    after_unload = print_runtime_memory("after model unload")
    try:
        state = _load_model_stack(model_id)
    except Exception:
        _release_active_model()
        if previous_id is not None:
            try:
                _STATE.model.update(_load_model_stack(previous_id))
            except Exception as rollback_error:
                raise RuntimeError(
                    f"Loading {model_id!r} failed and rollback to {previous_id!r} also failed."
                ) from rollback_error
        raise
    _STATE.model.update(state)
    adapters.refresh_adapters()
    after_load = print_runtime_memory(f"after loading {model_id}")
    _STATE.model_switch_history.append(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "from": previous_id,
            "to": model_id,
            "backend_versions": rt.backend_versions,
            "memory_before": before,
            "memory_after_unload": after_unload,
            "memory_after_load": after_load,
        }
    )
    return model_id


def unload_model() -> None:
    """Release the active model, patched variants, and reclaimable memory."""
    _runtime()
    _release_active_model()
