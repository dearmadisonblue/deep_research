from __future__ import annotations

import gc
import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import sha256_file
from .config import MAX_PATCHED_MODEL_CACHE, ModelId
from .registry import available_models
from .runtime import _STATE, _runtime, _runtime_or_none


def _current_model() -> ModelId | None:
    return _STATE.model.get("model_id")


@dataclass(frozen=True)
class AdapterRecord:
    id: str
    path: Path
    model_id: ModelId


@dataclass(frozen=True)
class AdapterUse:
    id: str
    strength: float = 1.0


AdapterStatKey = tuple[int, int]
AdapterCacheKey = tuple[ModelId, str]
AdapterSignature = tuple[ModelId, tuple[tuple[str, float, int, int], ...]]

ADAPTER_INDEX: dict[ModelId, dict[str, AdapterRecord]] = {"qwen": {}}
ADAPTER_ALIASES: dict[ModelId, dict[str, str]] = {"qwen": {}}
ADAPTER_FILE_CACHE: dict[
    AdapterCacheKey, tuple[AdapterStatKey, dict[str, Any], dict[str, Any] | None]
] = {}
ADAPTER_HASH_CACHE: dict[AdapterCacheKey, tuple[AdapterStatKey, str]] = {}
PATCHED_MODEL_CACHE: OrderedDict[AdapterSignature, Any] = OrderedDict()
DEFAULT_ADAPTERS: tuple[AdapterUse, ...] = ()
ADAPTER_CATALOG_STATE: dict[AdapterCacheKey, tuple[str, AdapterStatKey]] = {}


def _adapter_stat_key(path: Path) -> AdapterStatKey:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def refresh_adapters() -> None:
    """Rebuild the adapter index and invalidate stale cached entries."""
    paths = _runtime().paths
    indexes: dict[ModelId, dict[str, AdapterRecord]] = {"qwen": {}}
    aliases: dict[ModelId, dict[str, str]] = {"qwen": {}}
    current: dict[AdapterCacheKey, tuple[str, AdapterStatKey]] = {}
    for model_id in available_models():
        model_dir = paths.adapters_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        stems: dict[str, list[str]] = {}
        for path in sorted(model_dir.rglob("*.safetensors")):
            adapter_id = path.relative_to(model_dir).with_suffix("").as_posix()
            record = AdapterRecord(id=adapter_id, path=path, model_id=model_id)
            indexes[model_id][adapter_id] = record
            stems.setdefault(path.stem, []).append(adapter_id)
            current[(model_id, adapter_id)] = (str(path), _adapter_stat_key(path))
        aliases[model_id] = {
            stem: ids[0]
            for stem, ids in stems.items()
            if len(ids) == 1 and stem not in indexes[model_id]
        }

    changed = {
        key
        for key in set(ADAPTER_CATALOG_STATE) | set(current)
        if ADAPTER_CATALOG_STATE.get(key) != current.get(key)
    }
    ADAPTER_INDEX.clear()
    ADAPTER_INDEX.update(indexes)
    ADAPTER_ALIASES.clear()
    ADAPTER_ALIASES.update(aliases)
    ADAPTER_CATALOG_STATE.clear()
    ADAPTER_CATALOG_STATE.update(current)
    if changed:
        for key in changed:
            ADAPTER_FILE_CACHE.pop(key, None)
            ADAPTER_HASH_CACHE.pop(key, None)
        clear_adapter_cache(files=False, models=True)
        gc.collect()
        if _runtime().torch.cuda.is_available():
            _runtime().torch.cuda.empty_cache()


def _get_adapter(name: str) -> AdapterRecord:
    model_id = _current_model()
    if model_id is None:
        raise RuntimeError("Load a model before selecting adapters.")
    key = name.strip()
    key = ADAPTER_ALIASES[model_id].get(key, key)
    try:
        return ADAPTER_INDEX[model_id][key]
    except KeyError:
        raise KeyError(
            f"Adapter not found for {model_id!r}: {name!r}. "
            f"Put it under {_runtime().paths.adapters_dir / model_id} and run refresh_adapters()."
        ) from None


def _load_adapter_state(name: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record = _get_adapter(name)
    key = (record.model_id, record.id)
    stat_key = _adapter_stat_key(record.path)
    cached = ADAPTER_FILE_CACHE.get(key)
    if cached is not None and cached[0] == stat_key:
        return cached[1], cached[2]
    from safetensors import safe_open
    from safetensors.torch import load_file

    state = load_file(record.path, device="cpu")
    with safe_open(record.path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
    ADAPTER_FILE_CACHE[key] = (stat_key, state, metadata)
    ADAPTER_HASH_CACHE.pop(key, None)
    return state, metadata


def _adapter_signature(adapters: Sequence[AdapterUse]) -> AdapterSignature:
    model_id = _current_model()
    if model_id is None:
        raise RuntimeError("No active model.")
    items = tuple(
        (
            item.id,
            round(item.strength, 8),
            *_adapter_stat_key(_get_adapter(item.id).path),
        )
        for item in adapters
        if item.strength != 0
    )
    return model_id, items


def _model_for_adapters(adapters: Sequence[AdapterUse]) -> Any:
    """Select non-fused PEFT adapters on one resident quantized pipeline."""
    pipeline = _STATE.model["pipeline"]
    loaded = _STATE.model.setdefault("loaded_adapters", {})
    signature = _adapter_signature(adapters)
    if not signature[1]:
        if loaded:
            pipeline.disable_lora()
        return pipeline
    names: list[str] = []
    weights: list[float] = []
    try:
        for adapter_id, strength, size, mtime_ns in signature[1]:
            key = (adapter_id, size, mtime_ns)
            if key not in loaded:
                name = "adapter_" + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
                state, _metadata = _load_adapter_state(adapter_id)
                pipeline.load_lora_weights(state, adapter_name=name)
                loaded[key] = name
            names.append(loaded[key])
            weights.append(strength)
        pipeline.enable_lora()
        pipeline.set_adapters(names, adapter_weights=weights)
    except Exception as exc:
        clear_adapter_cache(files=False, models=True)
        raise RuntimeError(
            "Unable to load the selected Qwen 2.1 LoRA with Diffusers/PEFT. "
            "Use a compatible Qwen 2.1 adapter; adapters are never fused into INT8 weights."
        ) from exc
    PATCHED_MODEL_CACHE[signature] = tuple(names)
    PATCHED_MODEL_CACHE.move_to_end(signature)
    while len(PATCHED_MODEL_CACHE) > MAX_PATCHED_MODEL_CACHE:
        PATCHED_MODEL_CACHE.popitem(last=False)
    retained = {name for value in PATCHED_MODEL_CACHE.values() for name in value}
    expired = [key for key, name in loaded.items() if name not in retained]
    if expired:
        pipeline.delete_adapters([loaded[key] for key in expired])
        for key in expired:
            del loaded[key]
    return pipeline


def set_default_adapters(adapters: Sequence[tuple[str, float]] | None = None) -> None:
    """Set session-default adapters as identifier-and-strength pairs."""
    global DEFAULT_ADAPTERS
    DEFAULT_ADAPTERS = tuple(
        AdapterUse(_get_adapter(name).id, float(strength))
        for name, strength in (adapters or [])
    )


def _clear_default_adapters() -> None:
    global DEFAULT_ADAPTERS
    DEFAULT_ADAPTERS = ()


def clear_adapter_cache(*, files: bool = True, models: bool = True) -> None:
    """Clear cached adapter tensors and/or loaded PEFT adapters."""
    if models:
        pipeline = _STATE.model.get("pipeline")
        # A failed adapter injection can be partial even before bookkeeping.
        if pipeline is not None and getattr(pipeline.transformer, "peft_config", None):
            pipeline.unload_lora_weights()
        _STATE.model.pop("loaded_adapters", None)
        PATCHED_MODEL_CACHE.clear()
    if files:
        ADAPTER_FILE_CACHE.clear()
        ADAPTER_HASH_CACHE.clear()
    gc.collect()
    rt = _runtime_or_none()
    if rt is not None and rt.torch.cuda.is_available():
        rt.torch.cuda.empty_cache()


def show_adapter_cache() -> None:
    """Print the loaded, patched, and session-default adapter state."""
    print("Active model:", _current_model())
    print("CPU adapter files:", list(ADAPTER_FILE_CACHE))
    print("Cached adapter selections:", list(PATCHED_MODEL_CACHE))
    print("Session defaults:", [(item.id, item.strength) for item in DEFAULT_ADAPTERS])


def _adapter_sha256(name: str) -> str:
    record = _get_adapter(name)
    key = (record.model_id, record.id)
    stat_key = _adapter_stat_key(record.path)
    cached = ADAPTER_HASH_CACHE.get(key)
    if cached is None or cached[0] != stat_key:
        cached = (stat_key, sha256_file(record.path))
        ADAPTER_HASH_CACHE[key] = cached
    return cached[1]
