from __future__ import annotations

import gc
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

ADAPTER_INDEX: dict[ModelId, dict[str, AdapterRecord]] = {"klein": {}, "qwen": {}}
ADAPTER_ALIASES: dict[ModelId, dict[str, str]] = {"klein": {}, "qwen": {}}
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
    indexes: dict[ModelId, dict[str, AdapterRecord]] = {"klein": {}, "qwen": {}}
    aliases: dict[ModelId, dict[str, str]] = {"klein": {}, "qwen": {}}
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
        PATCHED_MODEL_CACHE.clear()
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
    state, metadata = _runtime().comfy_utils.load_torch_file(
        str(record.path), safe_load=True, return_metadata=True
    )
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
    if not adapters:
        return _STATE.model["model"]
    signature = _adapter_signature(adapters)
    cached = PATCHED_MODEL_CACHE.get(signature)
    if cached is not None:
        PATCHED_MODEL_CACHE.move_to_end(signature)
        return cached
    model = _STATE.model["model"]
    for adapter_id, strength, _size, _mtime_ns in signature[1]:
        state, metadata = _load_adapter_state(adapter_id)
        model, _ = _runtime().comfy_sd.load_lora_for_models(
            model, None, state, strength, 0.0, lora_metadata=metadata
        )
    PATCHED_MODEL_CACHE[signature] = model
    while len(PATCHED_MODEL_CACHE) > MAX_PATCHED_MODEL_CACHE:
        PATCHED_MODEL_CACHE.popitem(last=False)
    return model


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
    """Clear cached adapter tensors and/or patched model variants."""
    if models:
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
    print("Patched model variants:", list(PATCHED_MODEL_CACHE))
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
