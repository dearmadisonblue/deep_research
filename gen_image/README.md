# gen_image

This experimental branch replaces the backend with Hugging Face Diffusers and
supports **Qwen Image 2.1 only**. The public functions, their signatures, Python
workflows, PIL return values, asset/prompt indexing, ordered references, batch
seeds, saving controls, and logging controls remain available.

## Installation

Use Python 3.11+ on a CUDA machine with BF16 support (Ampere or newer).
Install this branch into your existing environment, including Colab:

```bash
pip install --upgrade "git+https://github.com/dearmadisonblue/deep_research.git@experimental/qwen21-diffusers#subdirectory=gen_image"
```

Diffusers is installed from the official GitHub default branch, without a
fixed commit or version, because the current PyPI release does not yet include
Qwen 2.1 support. All other dependencies are ordinary, unversioned package
requirements. Pip can reuse already-installed packages; this package does not
pin or request a replacement ML stack. Dependencies of the libraries themselves
can still impose their own compatibility requirements.

Installing this package also installs its GitHub Diffusers dependency; no
separate Diffusers installation command is needed. Restart an existing notebook
session after installation before importing the package again. There is no
automatic library installation during `init()`.

The installed Diffusers library must provide `QwenImage21Pipeline`,
`QwenImage21Transformer2DModel`, and `AutoencoderKLQwenImage21`; TorchAO must
support the checkpoint's saved `Int8Tensor` serialization. `init()` checks the
required imports and records actual installed package versions. It never
installs packages or clones a backend at runtime.
If installation changes a library already imported in a notebook, restart the
session before importing it again.

## Source layout

The public API is re-exported from `gen_image.__init__`. Implementation code is
divided by responsibility:

- `config` and `registry` define shared types, constants, and supported models.
- `runtime` owns initialized backend and session state.
- `diffusers_backend` assembles the pipeline from pre-quantized components.
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
        └── qwen/
```

A second, disposable directory contains Hugging Face's native download cache:

```text
temp/
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
    gen_image.load_model("qwen")
    image = gen_image.draw(portrait_workflow)
    image.show()


asyncio.run(main())
```

`init()` configures the package but does not download or load a model.
`load_model()` checks the persistent cache, downloads and verifies missing
files, and then loads the selected model. `draw()` never displays an image; it
returns a PIL image, leaving presentation to the caller.

## Qwen 2.1 checkpoint configuration

All model weights come from [Unsloth's Qwen-Image-2.1-FP8 repository](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8):

| Component | Published file | Loading / compute |
| --- | --- | --- |
| Transformer | `Qwen-Image-2.1-INT8.safetensors` | Reconstruct saved TorchAO `Int8Tensor` weights and scales |
| Text encoder | `Qwen-Image-2.1-text_encoder-FP8.safetensors` | Keep FP8 storage; upcast each saved FP8 layer to BF16 during forward |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` | BF16 |

The Unsloth repository contains standalone components, not a complete Diffusers
pipeline directory. The loader constructs model skeletons on meta, assigns
weights strictly, and assembles `QwenImage21Pipeline` with the scheduler and
processor configuration from [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).
Only configuration and tokenizer files are downloaded from Qwen; its dense
transformer and text encoder are never downloaded. Every downloaded file has a
pinned repository revision, size, and SHA-256. Invalid or incompatible
checkpoints fail explicitly, with no runtime weight-quantization fallback.

Unsloth's INT8 artifact uses dynamic **activation** quantization in its forward
path. Its weights are already quantized. The FP8 encoder uses BF16 arithmetic,
not FP8 matrix multiplications. These checkpoints differ from the previous
INT8 ConvRot weights, so identical seeds do not imply identical images.

Defaults remain 1024 × 1024, 25 steps, guidance 1.0, and `use_sampler("euler")`.
Sampling uses `FlowMatchEulerDiscreteScheduler`; other sampler names raise a
clear error. `available_models()` returns `("qwen",)` and Klein aliases are
unsupported on this branch. Qwen aliases such as `"qwen-image-2.1"` still work.

Up to ten ordered references are supported, with `<image1>` through `<image10>`
in the prompt. The first reference supplies the default canvas; `use_size()`
overrides it. Reference alpha is preserved for the VAE and composited over
white for the text encoder by the Qwen pipeline. RGBA output is supported when
transparency is requested explicitly.

## Memory and first GPU trial

By default all three components stay on the GPU. The saved weights total about
16.1 GiB, before activations, prompt embeddings, and KV caches. Leave additional
VRAM and host RAM available. The loader avoids dense weight initialization,
and VAE tiling is enabled to reduce decoding peaks.

To keep the transformer resident while moving the whole text encoder to the
GPU only during prompt encoding, set this before `load_model()`:

```python
import os

os.environ["GEN_IMAGE_TEXT_ENCODER_OFFLOAD"] = "1"
```

This reduces VRAM during denoising, but temporarily needs space for the encoder
while encoding each prompt, and transfers about 9 GB each time. The default is
`"0"`. Change it before loading or use `load_model("qwen", force_reload=True)`.
The loader never calls generic pipeline CPU offloading or casts the INT8
transformer to another dtype.

A minimal trial needs no assets or adapters:

```python
await gen_image.init("/data/image-generation", "/tmp/gen-image")
gen_image.load_model("qwen")
gen_image.doctor(require_model=True)
gen_image.set_random_seed(42)
image = gen_image.draw(lambda: "A red ceramic teapot on a wooden table")
display(image)  # In a notebook; image.show() in a desktop Python session.
```

Then try an edit with `use_image(image)`, guidance/negative prompts, your Qwen
2.1 adapters, and batching. The CPU tests cover checkpoint reconstruction,
architecture compatibility, FP8 layer hooks, real PEFT adapter switching,
workflow arguments, and existing API behavior. They passed in the existing
development test environment, which has Diffusers `0.41.0.dev0`; they do not
establish compatibility with the ordinary `0.40.0` release. A separate miniature
random-weight CPU trial also completed text-to-image and reference-image editing with RGBA
output, CFG, and KV caching. **Full-size CUDA inference and
image quality have not been tested in the development workspace.**

## Workflows and effects

A workflow is a zero-argument Python function that returns a prompt:

```python
Workflow = Callable[[], str | Prompt]
```

Calls such as
`use_image()` and `use_adapter()` follow a React-inspired effect pattern: they
record declarations in a context local to the current `draw()` call rather
than performing generation immediately.

```python
def portrait_workflow():
    gen_image.use_image("subject/front")
    gen_image.use_size(1024, 1280)
    gen_image.use_steps(25)
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
ModelId = Literal["qwen"]
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
    """Initialize paths, caches, Diffusers dependencies, and file indexes."""
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
    """Validate directories, Diffusers dependencies, CUDA availability, and model state."""
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

Adapter files are discovered beneath `root/models/adapters/qwen`. Use adapters
compatible with Diffusers' Qwen 2.1 LoRA loader. Existing files stay in the same
location, but a format accepted by the previous backend may require conversion.
Incompatible files raise an error. Adapters are loaded as separate PEFT branches
and are never fused or requantized into the INT8 transformer. At most two adapter
selections are retained; clearing the model cache unloads their PEFT weights.

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
def clear_adapter_cache(*, files: bool = True, models: bool = True) -> None:
    """Clear cached adapter tensors and/or loaded PEFT adapters."""
```

This does not delete persistent files.

```python
def show_adapter_cache() -> None:
    """Print the loaded, patched, and session-default adapter state."""
```

## Workflow-effect API

These functions must be called while `draw()` is executing a workflow.

```python
def use_image(
    asset: Asset | str | Image.Image, *, strength: float | None = None
) -> None:
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
the prompt, model, seeds, references, effects, output paths, backend versions,
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

A single image returned by `draw()` can be used directly as a reference in a
later workflow, even when saving is disabled:

```python
first = draw(original_workflow)


def photofy():
    use_image(first)
    return "Convert this image to a photorealistic style"


second = draw(photofy)
```

For a batch result, select one image from the returned list before passing it
to `use_image()`.

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

## Verification

```bash
pip install pytest
pytest gen_image/tests -q  # From the repository root.
pip check
```

Provenance logs use schema 2 and record Diffusers/component revisions and
quantization details instead of backend node information. `status()` reports
backend package versions. Backend-specific diagnostic fields have changed;
the public call signatures have not.

Verified model and configuration files live under `root/models/huggingface/files`
in repository/revision directories. Download verification markers include the
file modification time, so changed files are hashed again. The disposable
Hugging Face cache may be deleted between sessions; complete persistent files
are reused without contacting the Hub.
