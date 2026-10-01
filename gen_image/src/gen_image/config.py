from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias
from zoneinfo import ZoneInfo

ModelId: TypeAlias = Literal["qwen"]
AssetKind: TypeAlias = Literal["image", "video"]
SecretProvider: TypeAlias = Callable[[str], str | None]

QWEN_REFERENCE_RESOLUTION = 1024
MAX_PATCHED_MODEL_CACHE = 2
NYC_TIMEZONE = ZoneInfo("America/New_York")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
