from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from . import adapters
from .catalog import Asset
from .config import NYC_TIMEZONE, QWEN_REFERENCE_RESOLUTION
from .diagnostics import print_runtime_memory
from .provenance import (
    _build_log_payload,
    atomic_save_png,
    atomic_write_json,
    is_logging,
)
from .registry import MODEL_REGISTRY
from .runtime import _STATE, _runtime
from .workflows import ImageUse, Workflow, _DrawSpec, _resolve_workflow


def _reference_image(image_use: ImageUse) -> Image.Image:
    source = image_use.source
    image_context = (
        Image.open(source.path) if isinstance(source, Asset) else nullcontext(source)
    )
    with image_context as image:
        return ImageOps.exif_transpose(image).convert("RGBA").copy()


def _generate_images(spec: _DrawSpec) -> tuple[list[Image.Image], list[int]]:
    if {"model_id", "pipeline"} - _STATE.model.keys():
        raise RuntimeError("No model is loaded.")
    if spec.model_id != _STATE.model.get("model_id"):
        raise RuntimeError("The active model changed while resolving the workflow.")
    if spec.sampler != "euler":
        raise ValueError(
            "The Diffusers Qwen 2.1 backend currently supports use_sampler('euler') only."
        )
    pipeline = adapters._model_for_adapters(spec.adapters)
    references = [_reference_image(item) for item in spec.images] or None
    seeds = [(spec.seed + index) & 0xFFFFFFFFFFFFFFFF for index in range(spec.batch)]
    torch = _runtime().torch
    images: list[Image.Image] = []
    with torch.no_grad():
        for seed in seeds:
            result = pipeline(
                prompt=spec.prompt,
                image=references,
                # An empty unconditional prompt still enables CFG when requested.
                negative_prompt=spec.negative_prompt if spec.guidance > 1 else None,
                true_cfg_scale=spec.guidance,
                width=spec.width,
                height=spec.height,
                num_inference_steps=spec.steps,
                generator=torch.Generator(
                    device=pipeline._execution_device
                ).manual_seed(seed),
                output_resolution=QWEN_REFERENCE_RESOLUTION,
                use_kv_cache=True,
                output_type="pil",
            )
            if len(result.images) != 1 or not isinstance(result.images[0], Image.Image):
                raise RuntimeError("Qwen pipeline returned an unexpected image batch.")
            images.append(result.images[0])
    return images, seeds


def draw(workflow: Workflow) -> Image.Image | list[Image.Image]:
    """Resolve and execute a workflow using the active model."""
    spec = _resolve_workflow(workflow)
    model_cfg = MODEL_REGISTRY[spec.model_id]
    print(f"Model: {model_cfg['display_name']} ({spec.model_id})")
    print("Workflow:", spec.workflow_name)
    print(f"Size: {spec.width}x{spec.height}")
    print(f"Steps / guidance: {spec.steps} / {spec.guidance}")
    print("Sampler:", spec.sampler)
    print("References:", len(spec.images))
    if spec.model_id == "qwen" and spec.negative_prompt.strip() and spec.guidance == 1:
        print("Warning: Qwen negative prompting has little or no effect at CFG 1.")
    if spec.model_id == "qwen" and spec.images and spec.size_explicit:
        print("Warning: explicit Qwen edit size overrides the first reference canvas.")
    if spec.model_id == "qwen" and len(spec.images) > 1:
        missing_tokens = [
            f"<image{index}>"
            for index in range(1, len(spec.images) + 1)
            if f"<image{index}>" not in spec.prompt
        ]
        if missing_tokens:
            print(
                "Warning: multi-reference prompt does not mention:",
                ", ".join(missing_tokens),
            )
    print(
        "Adapters:",
        ", ".join(f"{item.id}@{item.strength:g}" for item in spec.adapters) or "none",
    )
    print("Batch:", spec.batch)
    print("Saving:", "enabled" if spec.saving_enabled else "disabled")

    rt = _runtime()
    if rt.torch.cuda.is_available():
        rt.torch.cuda.reset_peak_memory_stats()
    memory_before = print_runtime_memory("before generation")
    try:
        generated, seeds = _generate_images(spec)
    except Exception:
        print_runtime_memory("after failed generation")
        raise
    memory_after = print_runtime_memory("after generation")

    now = datetime.now(UTC)
    timestamp_slug = now.strftime("%Y%m%dT%H%M%S.%fZ")
    timestamp_nyc = now.astimezone(NYC_TIMEZONE)
    run_id = uuid4().hex
    run_dir = rt.paths.photos_dir / timestamp_nyc.strftime("%Y-%m-%d")
    output_paths: list[Path | None] = []
    for index, (image, seed) in enumerate(zip(generated, seeds), start=1):
        path: Path | None = None
        if spec.saving_enabled:
            suffix = f"-{index:02d}" if len(generated) > 1 else ""
            filename = f"{spec.output_name}-{spec.model_id}-{timestamp_slug}-{seed}{suffix}.png"
            path = atomic_save_png(image, run_dir / filename)
            print("Saved:", path)
        output_paths.append(path)

    if is_logging():
        payload = _build_log_payload(
            spec=spec,
            run_id=run_id,
            timestamp_utc=now.isoformat(),
            timestamp_nyc=timestamp_nyc.isoformat(),
            seeds=seeds,
            images=generated,
            output_paths=output_paths,
            memory_before=memory_before,
            memory_after=memory_after,
        )
        log_path = atomic_write_json(
            rt.paths.run_logs_dir / f"{timestamp_slug}-{run_id}.json", payload
        )
        print("Log:", log_path)
    return generated[0] if len(generated) == 1 else generated
