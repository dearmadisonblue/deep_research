from __future__ import annotations

import ast
from pathlib import Path

import gen_image

PACKAGE_ROOT = Path(__file__).parents[1]
SOURCE = PACKAGE_ROOT / "src" / "gen_image"

EXPECTED_MODULES = {
    "__init__.py",
    "adapters.py",
    "catalog.py",
    "comfy_backend.py",
    "config.py",
    "diagnostics.py",
    "downloads.py",
    "generation.py",
    "models.py",
    "provenance.py",
    "registry.py",
    "runtime.py",
    "workflows.py",
}


def test_source_compiles() -> None:
    for path in SOURCE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_package_module_layout() -> None:
    assert {path.name for path in SOURCE.glob("*.py")} == EXPECTED_MODULES


def test_public_api_exports_session_accessors() -> None:
    assert gen_image.set_logging is not None
    assert gen_image.is_logging is not None
    assert gen_image.set_saving is not None
    assert gen_image.is_saving is not None
    assert not hasattr(gen_image, "use_save")
