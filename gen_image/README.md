# gen_image

`gen_image` is a Python image-generation package built around two tested model
configurations. It manages persistent model files, ComfyUI initialization,
model switching, prompt and asset indexing, workflow execution, image saving,
and provenance logging.

The package uses ComfyUI internals directly. It does not launch the ComfyUI web
interface or API server. Instead, it configures ComfyUI as an in-process
backend, registers the required native nodes, and invokes those nodes from
Python. This provides a compact API but couples the package more closely to
ComfyUI's internal APIs than a conventional HTTP client.

## Installation

```bash
pip install "git+https://github.com/dearmadisonblue/deep_research.git#subdirectory=gen_image"
```

## Source layout

The public API is re-exported from `gen_image.__init__`. Implementation code is
divided by responsibility:

- `config` and `registry` define shared types, constants, and supported models.
- `runtime` owns initialized backend and session state.
- `comfy_backend` contains the direct ComfyUI integration.
- `downloads`, `models`, and `adapters` manage model resources and activation.
- `catalog` indexes prompts and assets.
- `workflows` records and resolves declarative workflow effects.
- `generation` performs conditioning, sampling, saving, and draw execution.
- `provenance` builds output records and controls provenance logging.
- `diagnostics` provides status, health, memory, and model-switch checks.

## Directory layout

`init()` receives a persistent root directory with the following layout:

```text
root/
├── assets/
├── prompts/
├── media/
│   ├── photos/
│   └── videos/
├── logs/
│   └── runs/
└── models/
    ├── huggingface/
    │   └── files/
    └── adapters/
        ├── klein/
        └── qwen/
```

A second, disposable directory contains the ComfyUI checkout and Hugging
Face's native download cache:

```text
temp/
├── ComfyUI/
└── huggingface/
    ├── hub/
    └── xet/
```

Complete, verified model files are persisted under `root/models`. Everything
under `temp` may be discarded between sessions.

## Quick start

```python
import asyncio
from pathlib import Path

import gen_image


def portrait_workflow():
    gen_image.use_image("references/portrait")
    gen_image.use_adapter("portrait-detail", 0.9)
    gen_image.use_output_name("portrait")
    return "A photorealistic editorial portrait in soft window light."


async def main():
    await gen_image.init(
        root_dir=Path("/data/image-generation"),
        temp_dir=Path("/tmp/gen-image"),
    )
    gen_image.load_model("klein")
    image = gen_image.draw(portrait_workflow)
    image.show()


asyncio.run(main())
```

`init()` configures the package but does not download or load a model.
`load_model()` checks the persistent cache, downloads and verifies missing
files, and then loads the selected model. `draw()` never displays an image; it
returns a PIL image, leaving presentation to the caller.

## Model configurations

### `klein`

- `flux-2-klein-9b-fp8.safetensors`
- `qwen_3_8b_fp8mixed.safetensors`
- `full_encoder_small_decoder.safetensors`
- Default: 1024 × 1024, 4 steps, guidance 1.0, Euler

This stack uses ComfyUI's native FLUX.2 conditioning, reference-latent,
scheduler, sampler, and VAE nodes.

### `qwen`

- `qwen_image_2.1_int8_convrot.safetensors`
- `qwen3vl_8b_int8_convrot.safetensors`
- `qwen_image_2.1_vae_bf16.safetensors`
- Default: 1024 × 1024, 25 steps, guidance 1.0, Euler/simple
- Up to ten ordered references

References are identified as `<image1>` through `<image10>`. For edits, the
first reference supplies the default canvas. RGBA output is supported when
transparency is requested explicitly.

## Workflows and effects

A workflow is a zero-argument Python function that returns a prompt:

```python
Workflow = Callable[[], str | Prompt]
```

It is not a ComfyUI node graph or serialized ComfyUI workflow. Calls such as
`use_image()` and `use_adapter()` follow a React-inspired effect pattern: they
record declarations in a context local to the current `draw()` call rather
than performing generation immediately.

```python
def portrait_workflow():
    gen_image.use_image("subject/front")
    gen_image.use_size(1024, 1280)
    gen_image.use_steps(6)
    return "A full-length studio portrait."
```

Resolution precedence is:

```text
active-model defaults → workflow effects
```

The context is isolated with `ContextVar`, so settings do not leak between
draws. Unlike React Hooks, effects may be conditional and do not participate in
a render or reconciliation cycle.

## Core types

```python
ModelId = Literal["klein", "qwen"]
AssetKind = Literal["image", "video"]
SecretProvider = Callable[[str], str | None]
Workflow = Callable[[], str | Prompt]


class Prompt(str):
    """Prompt text with a canonical ID, source path, and YAML metadata."""

    id: str
    path: Path
    metadata: Mapping[str, Any]
    source_sha256: str


@dataclass(frozen=True)
class Asset:
    """An indexed image or video file."""

    id: str
    path: Path
    kind: AssetKind

    def content_sha256(self) -> str:
        """Return the SHA-256 digest of the asset file."""
```

A single-image draw returns `PIL.Image.Image`. A batch draw returns
`list[PIL.Image.Image]`.

## Lifecycle API

```python
async def init(
    root_dir: str | Path,
    temp_dir: str | Path,
    *,
    secret_provider: SecretProvider | None = None,
) -> Mapping[str, Any]:
    """Initialize paths, caches, ComfyUI, native nodes, and file indexes."""
```

`root_dir` contains persistent data. `temp_dir` contains disposable backend
and download state. The secret provider is called only when an uncached gated
download requires authentication.

```python
def status() -> Mapping[str, Any]:
    """Return the current configuration, backend revision, and model state."""
```

`status()` reports state without modifying it.

```python
def doctor(*, require_model: bool = False) -> Mapping[str, Any]:
    """Validate directories, ComfyUI nodes, CUDA availability, and model state."""
```

`doctor()` returns a structured health report and raises if a required check
fails.

```python
def runtime_memory_snapshot() -> dict[str, Any]:
    """Return JSON-safe process, system-RAM, and CUDA memory measurements."""
```

```python
def print_runtime_memory(
    label: str,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Print and return a runtime memory snapshot."""
```

These functions expose the same measurements written to generation logs and
model-switch history.

## Model API

```python
def available_models() -> tuple[ModelId, ...]:
    """Return the supported model identifiers."""
```

```python
def model_spec(name: str | None = None) -> Mapping[str, Any]:
    """Return configuration metadata for a model or the active model."""
```

The specification includes components, defaults, quantization, reference
limits, and size constraints.

```python
def load_model(name: str, *, force_reload: bool = False) -> ModelId:
    """Download missing files, verify them, and activate a model stack."""
```

Target files are ready before the existing model is released. If loading
fails, the package attempts to restore the previous model.

```python
def current_model() -> ModelId | None:
    """Return the active model identifier, or None when none is loaded."""
```

```python
def unload_model() -> None:
    """Release the active model, patched variants, and reclaimable memory."""
```

Persistent model files remain on disk.

```python
def run_model_switch_smoke_test(*, restore: bool = True) -> Mapping[str, Any]:
    """Load every supported stack without sampling and report the result."""
```

When `restore=True`, the initially active model is restored afterward.

## Prompt and asset API

```python
def refresh_library() -> None:
    """Rebuild the in-memory prompt and asset indexes."""
```

Call this after adding, removing, or editing files.

```python
def get_prompt(prompt_id: str) -> Prompt:
    """Return a prompt by canonical ID or unambiguous filename stem."""
```

```python
def find_prompts(
    *,
    tags: str | Sequence[str] | None = None,
    where: Mapping[str, Any] | None = None,
) -> list[Prompt]:
    """Find prompts by tags and exact front-matter values."""
```

```python
def get_asset(asset_id: str) -> Asset:
    """Return an asset by relative ID or unambiguous filename stem."""
```

An asset's canonical ID is its extension-free path relative to `root/assets`.

```python
def find_assets(*, kind: AssetKind | None = None) -> list[Asset]:
    """Return indexed assets, optionally restricted by kind."""
```

Assets are indexed directly from their filenames and directory paths. They do
not have sidecar metadata.

## Adapter API

Adapter files are discovered beneath `root/models/adapters`. The model-specific
subdirectory determines compatibility.

```python
def refresh_adapters() -> None:
    """Rebuild the adapter index and invalidate stale cached entries."""
```

Call this after changing adapter files on disk.

```python
def use_adapter(name: str, strength: float = 1.0) -> None:
    """Apply an indexed adapter to the current workflow."""
```

The file is loaded lazily on the first relevant draw. Multiple calls compose
in declaration order.

```python
def set_default_adapters(
    adapters: Sequence[tuple[str, float]] | None = None,
) -> None:
    """Set session-default adapters as identifier-and-strength pairs."""
```

Passing `None` or an empty sequence restores the bare active model.

```python
def clear_adapter_cache(
    *, files: bool = True, models: bool = True
) -> None:
    """Clear cached adapter tensors and/or patched model variants."""
```

This does not delete persistent files.

```python
def show_adapter_cache() -> None:
    """Print the loaded, patched, and session-default adapter state."""
```

## Workflow-effect API

These functions must be called while `draw()` is executing a workflow.

```python
def use_image(asset: Asset | str, *, strength: float | None = None) -> None:
    """Add an ordered image reference to the current workflow."""

def use_adapter(name: str, strength: float = 1.0) -> None:
    """Apply an indexed adapter to the current workflow."""

def use_size(width: int, height: int) -> None:
    """Override the model's default output dimensions."""

def use_steps(steps: int) -> None:
    """Override the model's default sampling-step count."""

def use_guidance(guidance: float) -> None:
    """Override the model's default guidance value."""

def use_sampler(name: str) -> None:
    """Select the sampler for the current workflow."""

def use_negative_prompt(prompt: str | Prompt) -> None:
    """Set the negative prompt for the current workflow."""

def use_batch(batch: int) -> None:
    """Set the number of images generated by the current draw."""

def use_output_name(name: str) -> None:
    """Set the filename prefix used for saved images."""
```

Image references are ordered. Batch draws return one PIL image per batch item.

## Random-seed API

```python
def set_random_seed(seed: int | None) -> None:
    """Set a fixed generation seed, or restore automatic seeding with None."""
```

A fixed seed applies to every subsequent draw and survives model switches for
the lifetime of the Python process. Batch items use consecutive seeds starting
from the configured value. Passing `None` restores a fresh automatic seed for
each draw. This setting affects generation noise only; it does not seed Python,
NumPy, or Torch random-number generators.

## Saving API

```python
def set_saving(enabled: bool) -> None:
    """Enable or disable saving generated images to files."""

def is_saving() -> bool:
    """Return whether generated images are saved to files."""
```

Saving is enabled by default. The setting applies to subsequent draws and
survives model switches for the lifetime of the Python process. Disabling
saving does not affect the returned image or the logging setting.

## Logging API

```python
def set_logging(enabled: bool) -> None:
    """Enable or disable generation provenance logging."""

def is_logging() -> bool:
    """Return whether generation provenance logging is enabled."""
```

When enabled, each draw writes JSON provenance under `root/logs/runs`, including
the prompt, model, seeds, references, effects, output paths, backend revision,
and memory measurements.

## Execution API

```python
def draw(workflow: Workflow) -> Image.Image | list[Image.Image]:
    """Resolve and execute a workflow using the active model."""
```

`draw()` executes the workflow in an isolated effect context, generates the
requested image or batch, saves images under `root/media/photos` when enabled,
writes enabled provenance logs, and returns the generated PIL image or images.
It never displays or otherwise presents them.

## Prompt files

Prompt files are Markdown with YAML front matter:

```markdown
---
schema: 1
id: studio-portrait
tags: [portrait, studio]
subject: portrait
lighting: studio
---

A photorealistic studio portrait with soft directional lighting.
```

The front-matter `id` is canonical. An unambiguous filename stem is also an
alias. Front matter is used only to identify, organize, categorize, and find
prompts. It never configures inference; generation settings belong to the
active model and workflow effects.

## Asset IDs

`assets/characters/alice/front.jpg` receives the ID
`characters/alice/front`. If its filename stem is unique, it can also be
resolved as `front`.

## Backend behavior

The package configures ComfyUI for normal VRAM operation with dynamic VRAM,
asynchronous offload, smart memory, BF16 VAE operation, and previews disabled.

Model files are checked against expected byte counts and SHA-256 digests before
being accepted into persistent storage. Hugging Face's native cache remains
under `temp_dir`; only complete, verified files are copied into
`root_dir/models`.

Because normal ComfyUI startup is bypassed, the package explicitly registers
the native sampler, FLUX, editing, and Qwen nodes it requires. Its node invoker
supports both legacy ComfyUI call conventions and current V3
`execute()`/`NodeOutput` nodes.

The supported backend surface is deliberately narrow: the two documented model
stacks and their required native nodes.
