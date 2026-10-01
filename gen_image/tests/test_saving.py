from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from gen_image.workflows import _DrawSpec
from PIL import Image

import gen_image
from gen_image import adapters, generation, runtime


def test_saving_state_api() -> None:
    initial = gen_image.is_saving()
    try:
        gen_image.set_saving(False)
        assert gen_image.is_saving() is False
        gen_image.set_saving(True)
        assert gen_image.is_saving() is True
        with pytest.raises(TypeError):
            gen_image.set_saving(1)  # type: ignore[arg-type]
        assert gen_image.is_saving() is True
    finally:
        gen_image.set_saving(initial)


def test_draw_uses_saving_snapshot_and_logs_independently(
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
    image = Image.new("RGB", (1, 1))
    seen_specs: list[bool] = []

    def generate(spec: _DrawSpec) -> tuple[list[Image.Image], list[int]]:
        seen_specs.append(spec.saving_enabled)
        gen_image.set_saving(True)  # Change the global setting during this draw.
        return [image], [0]

    monkeypatch.setattr(generation, "_generate_images", generate)
    saved: list[Path] = []
    monkeypatch.setattr(
        generation, "atomic_save_png", lambda image, path: saved.append(path) or path
    )
    logged: list[tuple[bool, list[Path | None]]] = []

    def build_log(
        *, spec: _DrawSpec, output_paths: list[Path | None], **kwargs: object
    ) -> dict[str, object]:
        logged.append((spec.saving_enabled, output_paths))
        return {}

    monkeypatch.setattr(generation, "_build_log_payload", build_log)
    monkeypatch.setattr(generation, "atomic_write_json", lambda path, payload: path)

    initial_saving, initial_logging = gen_image.is_saving(), gen_image.is_logging()
    try:
        gen_image.set_saving(False)
        gen_image.set_logging(True)
        assert gen_image.draw(lambda: "a portrait") is image
        assert seen_specs == [False]
        assert saved == []
        assert logged == [(False, [None])]

        assert gen_image.draw(lambda: "a portrait") is image
        assert seen_specs == [False, True]
        assert len(saved) == 1
        assert logged[-1] == (True, saved)
    finally:
        gen_image.set_saving(initial_saving)
        gen_image.set_logging(initial_logging)
