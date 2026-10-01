"""Load Unsloth's published Qwen 2.1 components without runtime weight quantization."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .registry import BASE_MODEL_ID


async def initialize_backend(paths: Any) -> dict[str, Any]:
    import torch
    import torchvision

    # Import now so missing/incompatible dependencies fail at init, before a
    # multi-gigabyte checkpoint download. No source checkout or pip at runtime.
    from diffusers import (
        AutoencoderKLQwenImage21,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    _ = (
        torchvision,
        AutoencoderKLQwenImage21,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        unflatten_tensor_state_dict,
    )
    return {
        "torch": torch,
        "backend_versions": {
            **{
                name: version(name)
                for name in (
                    "torch",
                    "torchao",
                    "diffusers",
                    "transformers",
                    "accelerate",
                    "peft",
                    "torchvision",
                )
            },
        },
    }


def _load_prequantized(path: Path, *, component: str, scheme: str) -> dict[str, Any]:
    """Rebuild saved tensor subclasses, retaining their integer data and scales."""
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )

    expected_format = f"unsloth_prequant_{component}_state_dict_v1"
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        description = json.loads(metadata.get("unsloth_metadata", "{}"))
        if metadata.get("unsloth_format") != expected_format:
            raise ValueError(
                f"Unsupported {component} checkpoint format: {metadata.get('unsloth_format')!r}"
            )
        if (
            description.get("base_model_id") != BASE_MODEL_ID
            or description.get("scheme") != scheme
        ):
            raise ValueError(
                f"Checkpoint is not the expected {BASE_MODEL_ID} {scheme} {component}."
            )
        expected_class = (
            "QwenImage21Transformer2DModel"
            if component == "transformer"
            else "Qwen3VLForConditionalGeneration"
        )
        class_key = "transformer_class" if component == "transformer" else "te_class"
        if description.get(class_key) != expected_class:
            raise ValueError(
                f"Unexpected checkpoint model class: {description.get(class_key)!r}"
            )
        tensors = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
    state, leftover = unflatten_tensor_state_dict(tensors, metadata)
    expected_names = json.loads(metadata.get("tensor_names", "[]"))
    if leftover or not expected_names or set(state) != set(expected_names):
        raise ValueError(
            "Incomplete quantized checkpoint; refusing to fall back to dense weights."
        )
    if scheme == "int8":
        from torchao.quantization import Int8Tensor

        if not any(isinstance(value, Int8Tensor) for value in state.values()):
            raise ValueError("INT8 checkpoint contains no saved Int8Tensor weights.")
    else:
        import torch

        if not any(value.dtype == torch.float8_e4m3fn for value in state.values()):
            raise ValueError("FP8 checkpoint contains no FP8 weights.")
    return state


def _assign_weights(factory: Any, state: dict[str, Any]) -> Any:
    from accelerate import init_empty_weights

    # Leave nonpersistent buffers on CPU: they aren't in the saved state dict.
    # Never construct full-size randomly initialized dense model weights.
    with init_empty_weights(include_buffers=False):
        model = factory()
    model.load_state_dict(state, strict=True, assign=True)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Checkpoint assignment left meta tensors in the model.")
    model.requires_grad_(False)
    return model.eval()


def _arm_fp8_storage_hooks(encoder: Any) -> int:
    import torch
    from diffusers.hooks.layerwise_casting import apply_layerwise_casting_hook

    count = 0
    for name, module in encoder.named_modules():
        parameters = list(module.parameters(recurse=False))
        if not any(p.dtype == torch.float8_e4m3fn for p in parameters):
            continue
        # Hook only leaves already stored in FP8. Casting a mixed or enclosing
        # module would change the BF16 exclusions chosen by Unsloth.
        if list(module.children()) or any(
            p.dtype != torch.float8_e4m3fn for p in parameters
        ):
            raise ValueError(f"Unsupported mixed FP8 storage module: {name}")
        apply_layerwise_casting_hook(
            module, torch.float8_e4m3fn, torch.bfloat16, non_blocking=False
        )
        count += 1
    if not count:
        raise ValueError("No FP8 text-encoder layers found.")
    return count


def _stage_encoder(encoder: Any, device: Any) -> None:
    """Optional whole-encoder streaming; the quantized transformer stays resident."""

    def onload(module: Any, args: Any) -> None:
        module.to(device=device)  # Device only: preserve FP8 storage.

    def offload(module: Any, args: Any, output: Any) -> None:
        module.to(device="cpu")

    encoder.register_forward_pre_hook(onload)
    encoder.register_forward_hook(offload, always_call=True)


def load_pipeline(paths: dict[str, Path]) -> Any:
    import torch
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from safetensors.torch import load_file
    from transformers import (
        AutoConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen 2.1 INT8/FP8 requires a CUDA GPU with BF16 support.")
    device = torch.device("cuda")
    config_root = paths["pipeline_config"]
    transformer_config = QwenImage21Transformer2DModel.load_config(
        config_root / "transformer"
    )
    transformer = _assign_weights(
        lambda: QwenImage21Transformer2DModel.from_config(transformer_config),
        _load_prequantized(paths["diffusion"], component="transformer", scheme="int8"),
    )
    # .to(dtype=...) can destroy the checkpoint's TorchAO subclass semantics.
    transformer.to(device=device)
    encoder_config = AutoConfig.from_pretrained(
        config_root / "text_encoder", local_files_only=True
    )
    encoder = _assign_weights(
        lambda: Qwen3VLForConditionalGeneration(encoder_config),
        _load_prequantized(
            paths["text_encoder"], component="text_encoder", scheme="fp8"
        ),
    )
    _arm_fp8_storage_hooks(encoder)
    offload = os.environ.get("GEN_IMAGE_TEXT_ENCODER_OFFLOAD", "0") == "1"
    if offload:
        _stage_encoder(encoder, device)
    else:
        encoder.to(device=device)
    vae_config = AutoencoderKLQwenImage21.load_config(config_root / "vae")
    vae = _assign_weights(
        lambda: AutoencoderKLQwenImage21.from_config(vae_config),
        load_file(paths["vae"], device="cpu"),
    ).to(device=device, dtype=torch.bfloat16)
    vae.enable_tiling()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config_root / "scheduler", local_files_only=True
    )
    processor = Qwen3VLProcessor.from_pretrained(
        config_root / "processor", local_files_only=True
    )

    class _ResidentTransformerPipeline(QwenImage21Pipeline):
        @property
        def _execution_device(self) -> Any:
            # The text encoder may be on CPU between forwards. Sampling and
            # processor inputs always target the resident transformer's GPU.
            return self.transformer.device

    pipeline = _ResidentTransformerPipeline(
        transformer=transformer,
        text_encoder=encoder,
        vae=vae,
        scheduler=scheduler,
        processor=processor,
    )
    pipeline._gen_image_encoder_offload = offload
    # Don't call pipeline.to(dtype=...) or generic pipeline CPU-offload methods.
    return pipeline
