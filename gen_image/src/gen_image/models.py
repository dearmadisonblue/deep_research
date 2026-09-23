from __future__ import annotations

import gc
from datetime import UTC, datetime
from typing import Any

from . import adapters
from .comfy_backend import call_node
from .config import ModelId
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
    try:
        rt.model_management.unload_all_models()
    except Exception:
        rt.model_management.free_memory(1e32, rt.model_management.get_torch_device())
    _STATE.model.clear()
    gc.collect()
    try:
        rt.model_management.soft_empty_cache()
    except Exception:
        pass
    if rt.torch.cuda.is_available():
        rt.torch.cuda.empty_cache()
        rt.torch.cuda.synchronize()


def _load_model_stack(model_id: ModelId) -> dict[str, Any]:
    spec = MODEL_REGISTRY[model_id]
    paths = _ensure_model_files(model_id)
    components = spec["components"]
    model = call_node(
        "UNETLoader",
        unet_name=components["diffusion"]["filename"],
        weight_dtype="default",
    )[0]
    clip = call_node(
        "CLIPLoader",
        clip_name=components["text_encoder"]["filename"],
        type=spec["clip_type"],
        device="default",
    )[0]
    vae = call_node("VAELoader", vae_name=components["vae"]["filename"])[0]
    if model_id == "qwen":
        model = call_node(
            "QwenImage21Cache", model=model, device="auto", dtype="default"
        )[0]
    return {
        "model_id": model_id,
        "spec": spec,
        "paths": paths,
        "model": model,
        "clip": clip,
        "vae": vae,
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
            "comfy_git_revision": rt.comfy_git_revision,
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
