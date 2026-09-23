from __future__ import annotations

import pytest

import gen_image


def test_logging_state_api() -> None:
    initial = gen_image.is_logging()
    try:
        gen_image.set_logging(False)
        assert gen_image.is_logging() is False

        gen_image.set_logging(True)
        assert gen_image.is_logging() is True

        with pytest.raises(TypeError):
            gen_image.set_logging(1)  # type: ignore[arg-type]
    finally:
        gen_image.set_logging(initial)
