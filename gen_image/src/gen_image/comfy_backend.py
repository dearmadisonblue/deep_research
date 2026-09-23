from __future__ import annotations

import hashlib
import importlib
import math
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps

from .config import REFERENCE_MEGAPIXELS
from .registry import REQUIRED_NODES
from .runtime import _runtime

if TYPE_CHECKING:
    from .runtime import RuntimePaths


def _run_checked(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def _ensure_comfyui(paths: RuntimePaths) -> None:
    if not paths.comfy_dir.exists():
        paths.temp_dir.mkdir(parents=True, exist_ok=True)
        _run_checked(
            [
                "git",
                "clone",
                "https://github.com/Comfy-Org/ComfyUI.git",
                str(paths.comfy_dir),
            ]
        )
    elif not (paths.comfy_dir / ".git").exists():
        raise RuntimeError(
            f"ComfyUI directory is not a Git checkout: {paths.comfy_dir}"
        )

    requirements = paths.comfy_dir / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = paths.temp_dir / ".comfyui-requirements.sha256"
    if not marker.exists() or marker.read_text(encoding="utf-8").strip() != digest:
        _run_checked(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
        )
        marker.write_text(digest + "\n", encoding="utf-8")


async def initialize_backend(paths: RuntimePaths) -> dict[str, Any]:
    """Install and initialize the supported in-process ComfyUI backend."""
    _ensure_comfyui(paths)
    if str(paths.comfy_dir) not in sys.path:
        sys.path.insert(0, str(paths.comfy_dir))

    original_argv = sys.argv[:]
    sys.argv = [
        "comfyui",
        "--base-directory",
        str(paths.comfy_dir),
        "--enable-dynamic-vram",
        "--async-offload",
        "--bf16-vae",
        "--preview-method",
        "none",
        "--disable-auto-launch",
    ]
    try:
        comfy_options = importlib.import_module("comfy.options")
        comfy_options.enable_args_parsing()
        args = importlib.import_module("comfy.cli_args").args
        nodes = importlib.import_module("nodes")
        torch = importlib.import_module("torch")
        model_management = importlib.import_module("comfy.model_management")
        comfy_utils = importlib.import_module("comfy.utils")
        comfy_sd = importlib.import_module("comfy.sd")
        folder_paths = importlib.import_module("folder_paths")
    finally:
        sys.argv = original_argv

    folder_paths.add_model_folder_path("loras", str(paths.adapters_dir))
    required_extra_files = (
        paths.comfy_dir / "comfy_extras" / "nodes_custom_sampler.py",
        paths.comfy_dir / "comfy_extras" / "nodes_flux.py",
        paths.comfy_dir / "comfy_extras" / "nodes_edit_model.py",
        paths.comfy_dir / "comfy_extras" / "nodes_qwen.py",
    )
    for extra_file in required_extra_files:
        if not extra_file.exists():
            raise RuntimeError(
                f"Required native ComfyUI module is missing: {extra_file}"
            )
        if not await nodes.load_custom_node(
            str(extra_file), module_parent="comfy_extras"
        ):
            raise RuntimeError(f"Failed to register ComfyUI nodes from {extra_file}")

    mappings = nodes.NODE_CLASS_MAPPINGS
    missing = [name for name in REQUIRED_NODES if name not in mappings]
    if missing:
        raise RuntimeError(f"ComfyUI checkout lacks required native nodes: {missing}")

    revision = subprocess.check_output(
        ["git", "-C", str(paths.comfy_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "comfy_git_revision": revision,
        "torch": torch,
        "model_management": model_management,
        "comfy_utils": comfy_utils,
        "comfy_sd": comfy_sd,
        "folder_paths": folder_paths,
        "args": args,
        "node_class_mappings": mappings,
    }


def call_node(node_name: str, /, **kwargs: Any) -> tuple[Any, ...]:
    """Invoke a ComfyUI V1 or V3 node directly from Python."""
    mappings = _runtime().node_class_mappings
    if node_name not in mappings:
        raise KeyError(f"ComfyUI node not found: {node_name}")
    node_cls = mappings[node_name]
    if hasattr(node_cls, "define_schema") and hasattr(node_cls, "execute"):
        output = node_cls.execute(**kwargs)
    else:
        node = node_cls()
        output = getattr(node, node.FUNCTION)(**kwargs)
    if hasattr(output, "result"):
        output = output.result
    if output is None:
        return ()
    if isinstance(output, tuple):
        return output
    if isinstance(output, list):
        return tuple(output)
    return (output,)


def pil_to_comfy_image(image: Image.Image, *, preserve_alpha: bool = False) -> Any:
    rt = _runtime()
    image = ImageOps.exif_transpose(image)
    has_alpha = "A" in image.getbands() or (
        image.mode == "P" and "transparency" in image.info
    )
    image = image.convert("RGBA" if preserve_alpha and has_alpha else "RGB")
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    return rt.torch.from_numpy(array).unsqueeze(0)


def comfy_image_to_pil(image_tensor: Any) -> Image.Image:
    array = image_tensor.detach().cpu().numpy()
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise RuntimeError(f"Expected HWC RGB/RGBA tensor, got {tuple(array.shape)}")
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA" if array.shape[-1] == 4 else "RGB")


def scale_comfy_image_to_total_pixels(
    image: Any,
    *,
    megapixels: float = REFERENCE_MEGAPIXELS,
    upscale_method: str = "lanczos",
) -> Any:
    samples = image.movedim(-1, 1)
    total = megapixels * 1024 * 1024
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    scaled = _runtime().comfy_utils.common_upscale(
        samples, int(width), int(height), upscale_method, "disabled"
    )
    return scaled.movedim(1, -1)
