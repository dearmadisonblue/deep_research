from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from gen_image.diffusers_backend import (
    _arm_fp8_storage_hooks,
    _assign_weights,
    _load_prequantized,
)
from gen_image.registry import BASE_MODEL_ID
from gen_image.workflows import _resolve_workflow
from safetensors.torch import save_file
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
from torchao.quantization import Int8Tensor, MappingType, PerRow
from torchao.quantization.quantize_.workflows import QuantizeTensorToInt8Kwargs
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

import gen_image
from gen_image import adapters, generation, runtime

FIXTURES = Path(__file__).parent / "fixtures"


def _int8(weight: torch.Tensor) -> Int8Tensor:
    # Fixture creation only. Production loads published weights and never calls this.
    return Int8Tensor.from_hp(
        weight,
        PerRow(),
        act_quant_kwargs=QuantizeTensorToInt8Kwargs(PerRow(), MappingType.SYMMETRIC),
    )


def _checkpoint(
    path: Path,
    state: dict,
    *,
    component="transformer",
    scheme="int8",
    base=BASE_MODEL_ID,
) -> None:
    tensors, metadata = flatten_tensor_state_dict(state)
    metadata["unsloth_format"] = f"unsloth_prequant_{component}_state_dict_v1"
    metadata["unsloth_metadata"] = json.dumps(
        {
            "base_model_id": base,
            "scheme": scheme,
            "transformer_class": "QwenImage21Transformer2DModel",
            "te_class": "Qwen3VLForConditionalGeneration",
        }
    )
    save_file(tensors, path, metadata=metadata)


def test_saved_int8_reconstructs_and_executes_without_requantizing(tmp_path):
    original = _int8(torch.randn(16, 32, dtype=torch.bfloat16))
    path = tmp_path / "int8.safetensors"
    _checkpoint(path, {"0.weight": original})
    state = _load_prequantized(path, component="transformer", scheme="int8")
    model = _assign_weights(
        lambda: torch.nn.Sequential(torch.nn.Linear(32, 16, bias=False)), state
    )
    assert isinstance(model[0].weight, Int8Tensor)
    assert torch.equal(model[0].weight.qdata, original.qdata)
    assert torch.equal(model[0].weight.scale, original.scale)
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(
        model(inputs), torch.nn.functional.linear(inputs, original)
    )
    assert not any(p.is_meta for p in model.parameters())


def test_wrong_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "wrong.safetensors"
    _checkpoint(
        path,
        {"0.weight": _int8(torch.randn(16, 32, dtype=torch.bfloat16))},
        base="Qwen/Qwen-Image",
    )
    with pytest.raises(ValueError, match="not the expected"):
        _load_prequantized(path, component="transformer", scheme="int8")


def test_fp8_storage_computes_in_bf16_and_restores_storage(tmp_path):
    original = torch.randn(16, 32, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    path = tmp_path / "fp8.safetensors"
    _checkpoint(path, {"0.weight": original}, component="text_encoder", scheme="fp8")
    state = _load_prequantized(path, component="text_encoder", scheme="fp8")
    model = _assign_weights(
        lambda: torch.nn.Sequential(torch.nn.Linear(32, 16, bias=False)), state
    )
    assert _arm_fp8_storage_hooks(model) == 1
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(inputs, original.to(torch.bfloat16))
    torch.testing.assert_close(model(inputs), expected)
    assert model[0].weight.dtype == torch.float8_e4m3fn


def test_published_checkpoints_match_pinned_model_architectures():
    # Configs and shape fingerprints taken from the pinned public checkpoint
    # headers. All three full-size architectures are built on meta, no weights
    # downloaded or dense initialization performed.
    fixture = json.loads((FIXTURES / "qwen21_architecture.json").read_text())
    with init_empty_weights(include_buffers=False):
        models = {
            "transformer": QwenImage21Transformer2DModel.from_config(
                fixture["transformer"]["config"]
            ),
            "text_encoder": Qwen3VLForConditionalGeneration(
                Qwen3VLConfig.from_dict(fixture["text_encoder"]["config"])
            ),
            "vae": AutoencoderKLQwenImage21.from_config(fixture["vae"]["config"]),
        }
    for name, model in models.items():
        shapes = {key: list(t.shape) for key, t in model.state_dict().items()}
        assert len(shapes) == fixture[name]["parameter_count"]
        assert (
            hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest()
            == fixture[name]["shape_sha256"]
        )
        assert not any(b.is_meta for b in model.buffers())


class _Processor:
    tokenizer = SimpleNamespace(encode=lambda text: [1])

    def apply_chat_template(self, *args, **kwargs):
        return [[1, 2]]


def _tiny_transformer():
    return (
        QwenImage21Transformer2DModel(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=16,
            axes_dims_rope=(4, 6, 6),
        )
        .to(dtype=torch.bfloat16)
        .eval()
    )


def test_lora_switching_keeps_int8_weights_and_restores_bare_model(
    tmp_path, monkeypatch
):
    transformer = _tiny_transformer()
    projection = transformer.transformer_blocks[0].attn.to_q
    projection.weight = torch.nn.Parameter(
        _int8(projection.weight.detach()), requires_grad=False
    )
    saved_qdata = projection.weight.qdata.clone()
    pipeline = QwenImage21Pipeline(
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_encoder=None,
        processor=_Processor(),
        vae=None,
    )
    state = {"model_id": "qwen", "pipeline": pipeline}
    monkeypatch.setattr(runtime._STATE, "model", state)
    fake_runtime = SimpleNamespace(
        torch=torch, paths=SimpleNamespace(adapters_dir=tmp_path)
    )
    monkeypatch.setattr(adapters, "_runtime", lambda: fake_runtime)
    monkeypatch.setattr(adapters, "_runtime_or_none", lambda: fake_runtime)
    adapter_dir = tmp_path / "qwen"
    adapter_dir.mkdir()
    save_file(
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(
                2, 32
            ),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(
                32, 2
            ),
        },
        adapter_dir / "detail.safetensors",
    )
    adapters.refresh_adapters()
    args = dict(
        hidden_states=torch.randn(1, 4, 4, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(1, 3, 16, dtype=torch.bfloat16),
        timestep=torch.tensor([0.5], dtype=torch.bfloat16),
        img_shapes=[[(1, 2, 2)]],
        img_mask=torch.tensor([[False, False, False, True]]),
    )
    with torch.no_grad():
        baseline = transformer(**args).sample
        adapters._model_for_adapters([adapters.AdapterUse("detail", 0.7)])
        adapted = transformer(**args).sample
        assert not torch.equal(adapted, baseline)
        adapters._model_for_adapters([])
        torch.testing.assert_close(transformer(**args).sample, baseline)
        adapters._model_for_adapters([adapters.AdapterUse("detail", 0.3)])
        assert len(state["loaded_adapters"]) == 1
        adapters.clear_adapter_cache()
        torch.testing.assert_close(transformer(**args).sample, baseline)
    assert isinstance(transformer.transformer_blocks[0].attn.to_q.weight, Int8Tensor)
    assert torch.equal(
        transformer.transformer_blocks[0].attn.to_q.weight.qdata, saved_qdata
    )


def test_workflow_maps_references_cfg_and_wrapped_batch_seeds(monkeypatch):
    from PIL import Image

    class Pipeline:
        _execution_device = torch.device("cpu")

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGBA", (64, 96))])

    pipeline = Pipeline()
    monkeypatch.setattr(
        runtime._STATE, "model", {"model_id": "qwen", "pipeline": pipeline}
    )
    monkeypatch.setattr(runtime._STATE, "random_seed", 2**64 - 1)
    monkeypatch.setattr(adapters, "DEFAULT_ADAPTERS", ())
    monkeypatch.setattr(generation, "_runtime", lambda: SimpleNamespace(torch=torch))
    references = [
        Image.new("RGBA", (64, 96), (1, 2, 3, 42)),
        Image.new("RGB", (64, 96), "red"),
    ]

    def workflow():
        for image in references:
            gen_image.use_image(image)
        gen_image.use_size(64, 96)
        gen_image.use_guidance(3)
        gen_image.use_batch(2)
        return "Combine <image1> and <image2>"

    images, seeds = generation._generate_images(_resolve_workflow(workflow))
    assert len(images) == 2
    assert seeds == [2**64 - 1, 0]
    assert [call["generator"].initial_seed() for call in pipeline.calls] == seeds
    for call in pipeline.calls:
        assert call["negative_prompt"] == ""
        assert call["true_cfg_scale"] == 3
        assert (call["width"], call["height"]) == (64, 96)
        assert call["image"][0].getpixel((0, 0)) == (1, 2, 3, 42)
        assert call["image"][1].getpixel((0, 0)) == (255, 0, 0, 255)
        assert call["use_kv_cache"] is True
