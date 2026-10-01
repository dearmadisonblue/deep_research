from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .config import ModelId

BASE_MODEL_ID = "Qwen/Qwen-Image-2.1"
BASE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
DIFFUSERS_REVISION = "578c9b2c6636ab2424a0e56186268b83623656b2"

MODEL_ALIASES = {
    "qwen": "qwen",
    "qwen-image": "qwen",
    "qwen_image_2_1": "qwen",
    "qwen-image-2.1": "qwen",
}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "qwen": {
        "display_name": "Qwen Image 2.1 Unsloth INT8 / FP8",
        "quantization": "pre-quantized TorchAO INT8 transformer / FP8-storage text encoder (BF16 compute)",
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
                "filename": "Qwen-Image-2.1-INT8.safetensors",
                "repo_id": "unsloth/Qwen-Image-2.1-FP8",
                "revision": "d67caebb412f98c968f9be41b3a8cea80b3e039a",
                "repo_filename": "Qwen-Image-2.1-INT8.safetensors",
                "size": 7258361376,
                "sha256": "51ed9fe73c8780e43d91c2ff2f36bf609369a8af2761d617b9a10c83d6ea3c60",
            },
            "text_encoder": {
                "filename": "Qwen-Image-2.1-text_encoder-FP8.safetensors",
                "repo_id": "unsloth/Qwen-Image-2.1-FP8",
                "revision": "d67caebb412f98c968f9be41b3a8cea80b3e039a",
                "repo_filename": "Qwen-Image-2.1-text_encoder-FP8.safetensors",
                "size": 9394530592,
                "sha256": "0a1e217ea5a327c77cf4c58ee2ec4b15dabdd55cc4ffa6d999085865e70db3df",
            },
            "vae": {
                "filename": "qwen_image_2.1_vae_bf16.safetensors",
                "repo_id": "unsloth/Qwen-Image-2.1-FP8",
                "revision": "d67caebb412f98c968f9be41b3a8cea80b3e039a",
                "repo_filename": "vae/qwen_image_2.1_vae_bf16.safetensors",
                "size": 675508656,
                "sha256": "71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c",
            },
        },
    },
}


def available_models() -> tuple[ModelId, ...]:
    """Return the supported model identifiers."""
    return ("qwen",)


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
