from __future__ import annotations

import platform
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import psutil

from .adapters import ADAPTER_INDEX
from .catalog import ASSET_INDEX, PROMPT_INDEX
from .config import NYC_TIMEZONE
from .models import current_model, load_model
from .registry import available_models
from .runtime import _STATE, _runtime, _runtime_or_none


def runtime_memory_snapshot() -> dict[str, Any]:
    """Return JSON-safe process, system-RAM, and CUDA memory measurements."""
    gib = 1024**3
    now = datetime.now(UTC)
    virtual = psutil.virtual_memory()
    snapshot: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "timestamp_nyc": now.astimezone(NYC_TIMEZONE).isoformat(),
        "python_version": platform.python_version(),
        "process_rss_gib": round(psutil.Process().memory_info().rss / gib, 4),
        "ram_total_gib": round(virtual.total / gib, 4),
        "ram_available_gib": round(virtual.available / gib, 4),
        "ram_used_gib": round(virtual.used / gib, 4),
        "ram_percent": float(virtual.percent),
        "cuda_available": False,
    }
    rt = _runtime_or_none()
    if rt is not None:
        torch = rt.torch
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            snapshot["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_vram_gib": round(total_bytes / gib, 4),
                "free_vram_gib": round(free_bytes / gib, 4),
                "allocated_now_gib": round(torch.cuda.memory_allocated() / gib, 4),
                "reserved_now_gib": round(torch.cuda.memory_reserved() / gib, 4),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 4),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 4),
            }
    return snapshot


def print_runtime_memory(
    label: str, snapshot: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Print and return a runtime memory snapshot."""
    snapshot = snapshot or runtime_memory_snapshot()
    print(f"Memory — {label}")
    print(f"  process_rss_gib: {snapshot['process_rss_gib']:.2f}")
    print(f"  ram_available_gib: {snapshot['ram_available_gib']:.2f}")
    if gpu := snapshot.get("gpu"):
        print("  GPU:", gpu["name"])
        for key in (
            "total_vram_gib",
            "free_vram_gib",
            "allocated_now_gib",
            "reserved_now_gib",
            "peak_allocated_gib",
            "peak_reserved_gib",
        ):
            print(f"  {key}: {gpu[key]:.2f}")
    return snapshot


def status() -> Mapping[str, Any]:
    """Return the current configuration, backend revision, and model state."""
    rt = _runtime_or_none()
    if rt is None:
        return {"initialized": False, "active_model": None}
    return {
        "initialized": True,
        "root_dir": str(rt.paths.root_dir),
        "temp_dir": str(rt.paths.temp_dir),
        "backend": "diffusers",
        "backend_versions": rt.backend_versions,
        "active_model": current_model(),
        "prompt_count": len(PROMPT_INDEX),
        "asset_count": len(ASSET_INDEX),
        "adapter_count": sum(len(index) for index in ADAPTER_INDEX.values()),
    }


def doctor(*, require_model: bool = False) -> Mapping[str, Any]:
    """Validate directories, Diffusers dependencies, CUDA availability, and model state."""
    rt = _runtime()
    checks = {
        "root_directory_exists": rt.paths.root_dir.is_dir(),
        "temp_directory_exists": rt.paths.temp_dir.is_dir(),
        "models_directory_exists": rt.paths.models_dir.is_dir(),
        "cuda_available": bool(rt.torch.cuda.is_available()),
        "bf16_supported": bool(
            rt.torch.cuda.is_available() and rt.torch.cuda.is_bf16_supported()
        ),
        "backend_versions_recorded": bool(rt.backend_versions),
    }
    if require_model:
        checks["model_loaded"] = not ({"model_id", "pipeline"} - _STATE.model.keys())
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "ok" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "active_model": current_model(),
        "memory": runtime_memory_snapshot(),
    }
    if failures:
        raise RuntimeError("Health check failed: " + ", ".join(failures))
    return report


def run_model_switch_smoke_test(*, restore: bool = True) -> Mapping[str, Any]:
    """Load every supported stack without sampling and report the result."""
    starting_model = current_model()
    order = [model_id for model_id in available_models() if model_id != starting_model]
    if starting_model is not None:
        order.append(starting_model)
    loaded: list[str] = []
    try:
        for model_id in order:
            load_model(model_id)
            loaded.append(model_id)
    finally:
        if restore and starting_model is not None and current_model() != starting_model:
            load_model(starting_model)
    return {
        "status": "ok",
        "starting_model": starting_model,
        "loaded_in_order": loaded,
        "active_model": current_model(),
        "switch_events": len(_STATE.model_switch_history),
    }
