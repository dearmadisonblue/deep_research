from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gen_image import catalog, runtime


def _set_catalog_paths(monkeypatch, prompts: Path, assets: Path) -> None:
    monkeypatch.setattr(
        runtime,
        "_RUNTIME",
        SimpleNamespace(paths=SimpleNamespace(prompts_dir=prompts, assets_dir=assets)),
    )


def test_asset_ids_come_only_from_paths(monkeypatch, tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    assets = tmp_path / "assets"
    prompts.mkdir()
    (assets / "characters" / "alice").mkdir(parents=True)
    (assets / "characters" / "alice" / "front.jpg").write_bytes(b"test")
    (assets / "characters" / "alice" / "front.yaml").write_text(
        "id: ignored\n", encoding="utf-8"
    )
    _set_catalog_paths(monkeypatch, prompts, assets)

    catalog.refresh_library()

    asset = catalog.get_asset("characters/alice/front")
    assert asset.id == "characters/alice/front"
    assert asset.kind == "image"
    assert catalog.get_asset("front") == asset


def test_prompt_front_matter(monkeypatch, tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    assets = tmp_path / "assets"
    prompts.mkdir()
    assets.mkdir()
    (prompts / "portrait.md").write_text(
        """---
schema: 1
id: portrait
tags: [test]
category: portrait
---

A portrait.
""",
        encoding="utf-8",
    )
    _set_catalog_paths(monkeypatch, prompts, assets)

    catalog.refresh_library()

    prompt = catalog.get_prompt("portrait")
    assert str(prompt) == "A portrait."
    assert prompt.metadata["category"] == "portrait"
    assert catalog.find_prompts(tags="test") == [prompt]
