from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter

from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AssetKind
from .runtime import _runtime


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Prompt(str):
    """String-compatible prompt loaded from Markdown and YAML front matter."""

    def __new__(
        cls,
        text: str,
        *,
        prompt_id: str,
        metadata: Mapping[str, Any],
        path: Path,
        source_sha256: str,
    ) -> Prompt:
        obj = str.__new__(cls, text)
        obj.id = prompt_id
        obj.metadata = dict(metadata)
        obj.path = path
        obj.source_sha256 = source_sha256
        return obj

    id: str
    metadata: dict[str, Any]
    path: Path
    source_sha256: str


@dataclass(frozen=True)
class Asset:
    """An indexed image or video file."""

    id: str
    path: Path
    kind: AssetKind

    def content_sha256(self) -> str:
        return sha256_file(self.path)


PROMPT_INDEX: dict[str, Prompt] = {}
PROMPT_ALIASES: dict[str, str] = {}
ASSET_INDEX: dict[str, Asset] = {}
ASSET_ALIASES: dict[str, str] = {}


def _normalize_tags(value: Any, *, source: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise TypeError(f"tags must be a string or list[str]: {source}")


def _nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _matches_where(
    metadata: Mapping[str, Any], where: Mapping[str, Any] | None
) -> bool:
    return not where or all(
        _nested_get(metadata, key) == value for key, value in where.items()
    )


def _matches_tags(
    metadata: Mapping[str, Any], tags: str | Sequence[str] | None
) -> bool:
    if tags is None:
        return True
    required = [tags] if isinstance(tags, str) else list(tags)
    actual = set(_normalize_tags(metadata.get("tags"), source=Path("<metadata>")))
    return all(tag in actual for tag in required)


def refresh_library() -> None:
    """Rebuild the in-memory prompt and asset indexes."""
    paths = _runtime().paths
    prompt_index: dict[str, Prompt] = {}
    prompt_stems: dict[str, list[str]] = {}
    for path in sorted(paths.prompts_dir.rglob("*.md")):
        raw = path.read_bytes()
        post = frontmatter.loads(raw.decode("utf-8"))
        metadata = dict(post.metadata)
        if metadata.get("schema", 1) != 1:
            raise ValueError(f"Unsupported prompt schema in {path}")
        prompt_id = metadata.get("id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"Prompt requires a non-empty YAML id: {path}")
        prompt_id = prompt_id.strip()
        metadata["schema"] = 1
        metadata["id"] = prompt_id
        metadata["tags"] = _normalize_tags(metadata.get("tags"), source=path)
        if prompt_id in prompt_index:
            raise ValueError(f"Duplicate prompt id {prompt_id!r}")
        prompt_index[prompt_id] = Prompt(
            post.content.strip(),
            prompt_id=prompt_id,
            metadata=metadata,
            path=path,
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )
        prompt_stems.setdefault(path.stem, []).append(prompt_id)

    prompt_aliases = {
        stem: ids[0]
        for stem, ids in prompt_stems.items()
        if len(ids) == 1 and stem not in prompt_index
    }
    asset_index: dict[str, Asset] = {}
    asset_stems: dict[str, list[str]] = {}
    for path in sorted(
        candidate for candidate in paths.assets_dir.rglob("*") if candidate.is_file()
    ):
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            kind: AssetKind = "image"
        elif suffix in VIDEO_EXTENSIONS:
            kind = "video"
        else:
            continue
        asset_id = path.relative_to(paths.assets_dir).with_suffix("").as_posix()
        if asset_id in asset_index:
            raise ValueError(f"Duplicate asset id {asset_id!r}")
        asset_index[asset_id] = Asset(id=asset_id, path=path, kind=kind)
        asset_stems.setdefault(path.stem, []).append(asset_id)
    asset_aliases = {
        stem: ids[0]
        for stem, ids in asset_stems.items()
        if len(ids) == 1 and stem not in asset_index
    }
    PROMPT_INDEX.clear()
    PROMPT_INDEX.update(prompt_index)
    PROMPT_ALIASES.clear()
    PROMPT_ALIASES.update(prompt_aliases)
    ASSET_INDEX.clear()
    ASSET_INDEX.update(asset_index)
    ASSET_ALIASES.clear()
    ASSET_ALIASES.update(asset_aliases)
    print(f"Indexed {len(PROMPT_INDEX)} prompts and {len(ASSET_INDEX)} assets.")


def get_prompt(prompt_id: str) -> Prompt:
    """Return a prompt by canonical ID or unambiguous filename stem."""
    key = PROMPT_ALIASES.get(prompt_id.strip(), prompt_id.strip())
    try:
        return PROMPT_INDEX[key]
    except KeyError:
        raise KeyError(
            f"Prompt not found: {prompt_id!r}. Run refresh_library()."
        ) from None


def find_prompts(
    *,
    tags: str | Sequence[str] | None = None,
    where: Mapping[str, Any] | None = None,
) -> list[Prompt]:
    """Find prompts by tags and exact front-matter values."""
    return [
        prompt
        for prompt in sorted(PROMPT_INDEX.values(), key=lambda item: item.id)
        if _matches_tags(prompt.metadata, tags)
        and _matches_where(prompt.metadata, where)
    ]


def get_asset(asset_id: str) -> Asset:
    """Return an asset by relative ID or unambiguous filename stem."""
    key = ASSET_ALIASES.get(asset_id.strip(), asset_id.strip())
    try:
        return ASSET_INDEX[key]
    except KeyError:
        raise KeyError(
            f"Asset not found: {asset_id!r}. Run refresh_library()."
        ) from None


def find_assets(*, kind: AssetKind | None = None) -> list[Asset]:
    """Return indexed assets, optionally restricted by kind."""
    return [
        asset
        for asset in sorted(ASSET_INDEX.values(), key=lambda item: item.id)
        if kind is None or asset.kind == kind
    ]
