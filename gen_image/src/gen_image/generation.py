from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from . import adapters
from .comfy_backend import (
    call_node,
    comfy_image_to_pil,
    pil_to_comfy_image,
    scale_comfy_image_to_total_pixels,
)
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
from .workflows import Workflow, _DrawSpec, _resolve_workflow


def _prepare_klein_conditioning(spec: _DrawSpec) -> tuple[Any, Any, None]:
    clip, vae = _STATE.model["clip"], _STATE.model["vae"]
    positive = call_node("CLIPTextEncode", clip=clip, text=spec.prompt)[0]
    negative = (
        call_node("CLIPTextEncode", clip=clip, text=spec.negative_prompt)[0]
        if spec.negative_prompt.strip()
        else call_node("ConditioningZeroOut", conditioning=positive)[0]
    )
    for image_use in spec.images:
        with Image.open(image_use.asset.path) as image:
            tensor = pil_to_comfy_image(image, preserve_alpha=False)
        scaled = scale_comfy_image_to_total_pixels(tensor)
        latent = call_node("VAEEncode", pixels=scaled, vae=vae)[0]
        positive = call_node("ReferenceLatent", conditioning=positive, latent=latent)[0]
        negative = call_node("ReferenceLatent", conditioning=negative, latent=latent)[0]
    return positive, negative, None


def _prepare_qwen_conditioning(spec: _DrawSpec) -> tuple[Any, Any, Any]:
    images: dict[str, Any] = {}
    for index, image_use in enumerate(spec.images, start=1):
        with Image.open(image_use.asset.path) as image:
            images[f"image_{index}"] = pil_to_comfy_image(image, preserve_alpha=True)
    return call_node(
        "TextEncodeQwenImage21",
        clip=_STATE.model["clip"],
        prompt=spec.prompt,
        negative_prompt=spec.negative_prompt,
        vae=_STATE.model["vae"],
        resolution=QWEN_REFERENCE_RESOLUTION,
        images=images,
    )  # type: ignore[return-value]


def _generate_klein(spec: _DrawSpec, model: Any, seeds: list[int]) -> list[Image.Image]:
    positive, negative, _ = _prepare_klein_conditioning(spec)
    guider = call_node(
        "CFGGuider",
        model=model,
        positive=positive,
        negative=negative,
        cfg=spec.guidance,
    )[0]
    sampler = call_node("KSamplerSelect", sampler_name=spec.sampler)[0]
    sigmas = call_node(
        "Flux2Scheduler", steps=spec.steps, width=spec.width, height=spec.height
    )[0]
    images: list[Image.Image] = []
    for seed in seeds:
        noise = call_node("RandomNoise", noise_seed=seed)[0]
        latent = call_node(
            "EmptyFlux2LatentImage", width=spec.width, height=spec.height, batch_size=1
        )[0]
        sampled = call_node(
            "SamplerCustomAdvanced",
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )[0]
        decoded = call_node("VAEDecode", samples=sampled, vae=_STATE.model["vae"])[0]
        if decoded.ndim != 4 or decoded.shape[0] != 1:
            raise RuntimeError(f"Unexpected decoded tensor: {tuple(decoded.shape)}")
        images.append(comfy_image_to_pil(decoded[0]))
    return images


def _generate_qwen(spec: _DrawSpec, model: Any, seeds: list[int]) -> list[Image.Image]:
    positive, negative, edit_latent = _prepare_qwen_conditioning(spec)
    images: list[Image.Image] = []
    for seed in seeds:
        latent = (
            edit_latent
            if spec.images and not spec.size_explicit
            else call_node(
                "EmptyLatentImage", width=spec.width, height=spec.height, batch_size=1
            )[0]
        )
        sampled = call_node(
            "KSampler",
            model=model,
            seed=seed,
            steps=spec.steps,
            cfg=spec.guidance,
            sampler_name=spec.sampler,
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0,
        )[0]
        decoded = call_node("VAEDecode", samples=sampled, vae=_STATE.model["vae"])[0]
        if decoded.ndim != 4 or decoded.shape[0] != 1:
            raise RuntimeError(f"Unexpected decoded tensor: {tuple(decoded.shape)}")
        images.append(comfy_image_to_pil(decoded[0]))
    return images


def _generate_images(spec: _DrawSpec) -> tuple[list[Image.Image], list[int]]:
    if {"model_id", "model", "clip", "vae"} - _STATE.model.keys():
        raise RuntimeError("No model is loaded.")
    if spec.model_id != _STATE.model.get("model_id"):
        raise RuntimeError("The active model changed while resolving the workflow.")
    model = adapters._model_for_adapters(spec.adapters)
    seeds = [(spec.seed + index) & 0xFFFFFFFFFFFFFFFF for index in range(spec.batch)]
    with _runtime().torch.no_grad():
        if spec.model_id == "klein":
            return _generate_klein(spec, model, seeds), seeds
        return _generate_qwen(spec, model, seeds), seeds


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
