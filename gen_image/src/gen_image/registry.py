from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .config import ModelId

MODEL_ALIASES = {
    "klein": "klein",
    "flux": "klein",
    "flux2-klein": "klein",
    "qwen": "qwen",
    "qwen-image": "qwen",
    "qwen_image_2_1": "qwen",
    "qwen-image-2.1": "qwen",
}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "klein": {
        "display_name": "FLUX.2 Klein 9B FP8",
        "clip_type": "flux2",
        "quantization": "FP8 diffusion / FP8-mixed text encoder",
        "license": "Black Forest Labs model terms",
        "size_multiple": 16,
        "max_references": None,
        "defaults": {
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 1.0,
            "sampler": "euler",
            "negative_prompt": "",
            "batch": 1,
        },
        "components": {
            "diffusion": {
                "category": "diffusion_models",
                "filename": "flux-2-klein-9b-fp8.safetensors",
                "repo_id": "black-forest-labs/FLUX.2-klein-9b-fp8",
                "repo_filename": "flux-2-klein-9b-fp8.safetensors",
                "size": 9_433_061_528,
                "sha256": "865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee",
            },
            "text_encoder": {
                "category": "text_encoders",
                "filename": "qwen_3_8b_fp8mixed.safetensors",
                "repo_id": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
                "repo_filename": "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
                "size": 8_664_848_742,
                "sha256": "abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6",
            },
            "vae": {
                "category": "vae",
                "filename": "full_encoder_small_decoder.safetensors",
                "repo_id": "black-forest-labs/FLUX.2-small-decoder",
                "repo_filename": "full_encoder_small_decoder.safetensors",
                "size": 249_519_092,
                "sha256": "ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62",
            },
        },
    },
    "qwen": {
        "display_name": "Qwen Image 2.1 INT8 ConvRot",
        "clip_type": "qwen_image",
        "quantization": "INT8 ConvRot diffusion / INT8 ConvRot text encoder",
        "license": "Qwen Research License (non-commercial without a separate license)",
        "size_multiple": 32,
        "max_references": 10,
        "supports_rgba": True,
        "defaults": {
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "guidance": 1.0,
            "sampler": "euler",
            "negative_prompt": "",
            "batch": 1,
        },
        "components": {
            "diffusion": {
                "category": "diffusion_models",
                "filename": "qwen_image_2.1_int8_convrot.safetensors",
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "repo_filename": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
                "size": 7_256_783_064,
                "sha256": "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d",
            },
            "text_encoder": {
                "category": "text_encoders",
                "filename": "qwen3vl_8b_int8_convrot.safetensors",
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "repo_filename": "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                "size": 9_350_798_360,
                "sha256": "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f",
            },
            "vae": {
                "category": "vae",
                "filename": "qwen_image_2.1_vae_bf16.safetensors",
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "repo_filename": "vae/qwen_image_2.1_vae_bf16.safetensors",
                "size": 675_509_688,
                "sha256": "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
            },
        },
    },
}

KLEIN_REQUIRED_NODES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "RandomNoise",
    "CFGGuider",
    "KSamplerSelect",
    "Flux2Scheduler",
    "EmptyFlux2LatentImage",
    "SamplerCustomAdvanced",
    "VAEEncode",
    "VAEDecode",
    "ReferenceLatent",
}
QWEN_REQUIRED_NODES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "TextEncodeQwenImage21",
    "QwenImage21Cache",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
}
REQUIRED_NODES = sorted(KLEIN_REQUIRED_NODES | QWEN_REQUIRED_NODES)


def available_models() -> tuple[ModelId, ...]:
    """Return the supported model identifiers."""
    return ("klein", "qwen")


def normalize_model_id(name: str) -> ModelId:
    key = str(name).strip().lower()
    try:
        return MODEL_ALIASES[key]  # type: ignore[return-value]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Choose one of: {sorted(MODEL_REGISTRY)}"
        ) from None


def model_spec(name: str | None = None) -> Mapping[str, Any]:
    """Return configuration metadata for a model or the active model."""
    if name is None:
        from .runtime import _STATE

        model_id = _STATE.model.get("model_id")
    else:
        model_id = normalize_model_id(name)
    if model_id is None:
        raise RuntimeError("No model is active; provide a model name.")
    return copy.deepcopy(MODEL_REGISTRY[model_id])
