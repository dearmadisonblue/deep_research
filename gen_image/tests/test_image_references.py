from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from gen_image.catalog import Asset
from gen_image.provenance import _reference_record
from gen_image.workflows import _DrawSpec, _resolve_workflow
from PIL import Image

import gen_image
from gen_image import adapters, generation, runtime


def test_pil_reference_resolves_and_reaches_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "qwen"
    monkeypatch.setattr(runtime._STATE, "model", {"model_id": model_id})
    monkeypatch.setattr(adapters, "DEFAULT_ADAPTERS", ())
    reference = Image.new("RGBA", (400, 200), (255, 0, 0, 128))

    def workflow() -> str:
        gen_image.use_image(reference)
        reference.putpixel((0, 0), (0, 0, 255, 255))
        return "Edit this image"

    spec = _resolve_workflow(workflow)
    assert spec.images[0].source is not reference
    assert abs(spec.width - 2 * spec.height) <= 32
    prepared = generation._reference_image(spec.images[0])
    assert prepared.mode == "RGBA"
    assert prepared.getpixel((0, 0)) == (255, 0, 0, 128)

    record = _reference_record(spec.images[0], 1, model_id)
    assert record["id"] is None
    assert record["path"] is None
    assert len(record["sha256"]) == 64


def test_asset_reference_still_opens_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "reference.png"
    Image.new("RGB", (10, 10), "red").save(path)
    asset = Asset("reference", path, "image")
    prepared = generation._reference_image(generation.ImageUse(source=asset))
    assert prepared.getpixel((0, 0)) == (255, 0, 0, 255)
    assert prepared.mode == "RGBA"


def test_use_image_rejects_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime._STATE, "model", {"model_id": "qwen"})

    def workflow() -> str:
        gen_image.use_image([Image.new("RGB", (1, 1))])  # type: ignore[arg-type]
        return "Edit"

    with pytest.raises(TypeError, match="PIL image"):
        _resolve_workflow(workflow)


def test_draw_result_can_be_used_without_saving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime._STATE, "model", {"model_id": "qwen"})
    monkeypatch.setattr(adapters, "DEFAULT_ADAPTERS", ())
    fake_runtime = SimpleNamespace(
        paths=SimpleNamespace(
            photos_dir=tmp_path / "photos", run_logs_dir=tmp_path / "logs"
        ),
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(generation, "_runtime", lambda: fake_runtime)
    monkeypatch.setattr(generation, "print_runtime_memory", lambda label: {})
    first_image = Image.new("RGB", (100, 50), "red")
    seen: list[Image.Image] = []

    def generate(spec: _DrawSpec) -> tuple[list[Image.Image], list[int]]:
        if spec.images:
            assert isinstance(spec.images[0].source, Image.Image)
            seen.append(spec.images[0].source)
        return [first_image], [0]

    monkeypatch.setattr(generation, "_generate_images", generate)
    saving, logging = gen_image.is_saving(), gen_image.is_logging()
    try:
        gen_image.set_saving(False)
        gen_image.set_logging(False)
        image = gen_image.draw(lambda: "A portrait")

        def edit() -> str:
            gen_image.use_image(image)
            return "Make it photorealistic"

        gen_image.draw(edit)
    finally:
        gen_image.set_saving(saving)
        gen_image.set_logging(logging)

    assert len(seen) == 1
    assert seen[0].getpixel((0, 0)) == (255, 0, 0)
    assert not (tmp_path / "photos").exists()
