from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from vercel.blob.errors import BlobNotFoundError

from pickleball_vision.config import PersistenceSettings
from pickleball_vision.errors import PersistenceValidationError
from pickleball_vision.persistence.artifacts import (
    LocalArtifactStore,
    VercelBlobArtifactStore,
    create_artifact_store,
)
from pickleball_vision.persistence.models import (
    ArtifactAccess,
    ArtifactCategory,
    ArtifactPutRequest,
)


def token_factory(*tokens: str) -> Callable[[], str]:
    values = iter(tokens)
    return lambda: next(values)


@dataclass
class FakePutResult:
    pathname: str
    url: str
    content_type: str


@dataclass
class FakeGetResult:
    status_code: int
    stream: AsyncIterator[bytes] | None


@dataclass
class FakeHeadResult:
    pathname: str


class FakeBlobClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.urls: dict[str, str] = {}
        self.put_options: dict[str, object] = {}

    async def put(
        self,
        path: str,
        body: object,
        *,
        access: str,
        content_type: str | None,
        add_random_suffix: bool,
        overwrite: bool,
        multipart: bool,
    ) -> FakePutResult:
        data = bytearray()
        async for chunk in cast(AsyncIterator[bytes], body):
            data.extend(chunk)
        returned_path = f"{path}-provider-suffix" if add_random_suffix else path
        url = f"https://store.{access}.blob.vercel-storage.com/{returned_path}"
        self.objects[returned_path] = bytes(data)
        self.urls[url] = returned_path
        self.put_options = {
            "access": access,
            "contentType": content_type,
            "addRandomSuffix": add_random_suffix,
            "overwrite": overwrite,
            "multipart": multipart,
        }
        return FakePutResult(returned_path, url, content_type or "application/octet-stream")

    async def get(self, url_or_path: str, *, access: str) -> FakeGetResult | None:
        del access
        pathname = self.urls.get(url_or_path, url_or_path)
        data = self.objects.get(pathname)
        if data is None:
            return None

        async def chunks() -> AsyncIterator[bytes]:
            yield data

        return FakeGetResult(200, chunks())

    async def delete(self, url_or_path: str) -> None:
        pathname = self.urls.get(url_or_path, url_or_path)
        self.objects.pop(pathname, None)

    async def head(self, url_or_path: str) -> FakeHeadResult:
        pathname = self.urls.get(url_or_path, url_or_path)
        if pathname not in self.objects:
            raise BlobNotFoundError()
        return FakeHeadResult(pathname)


def test_local_artifact_store_round_trip_is_atomic_and_randomized(tmp_path: Path) -> None:
    source = tmp_path / "source match.mp4"
    source.write_bytes(b"private-media")
    store = LocalArtifactStore(
        tmp_path / "artifacts",
        token_factory=token_factory("put-token", "get-token"),
    )

    artifact = asyncio.run(
        store.put(
            ArtifactPutRequest(
                source_path=source,
                artifact_type="source_video",
                category=ArtifactCategory.SOURCE_MEDIA,
                match_id="match-1",
                pipeline_version="0.1.0",
            )
        )
    )

    assert artifact.access is ArtifactAccess.LOCAL
    assert artifact.url is None
    assert artifact.pathname == "source_media/match-1/put-token/source-match.mp4"
    assert artifact.checksum_sha256 is not None
    assert asyncio.run(store.exists(artifact)) is True

    restored = asyncio.run(store.get(artifact, tmp_path / "restored" / "source.mp4"))
    assert restored.read_bytes() == b"private-media"
    asyncio.run(store.delete(artifact))
    assert asyncio.run(store.exists(artifact)) is False
    assert not list((tmp_path / "artifacts").rglob("*.partial"))


def test_vercel_blob_adapter_uses_streams_and_explicit_public_viewable_access(
    tmp_path: Path,
) -> None:
    source = tmp_path / "annotated.mp4"
    source.write_bytes(b"viewable-media")
    client = FakeBlobClient()
    store = VercelBlobArtifactStore(
        "server-only-secret",
        client=client,
        token_factory=token_factory("put-token", "get-token"),
    )

    artifact = asyncio.run(
        store.put(
            ArtifactPutRequest(
                source_path=source,
                artifact_type="annotated_video",
                category=ArtifactCategory.VIEWABLE_MEDIA,
                access=ArtifactAccess.PUBLIC,
                match_id="match-1",
                content_type="video/mp4",
            )
        )
    )

    assert artifact.access is ArtifactAccess.PUBLIC
    assert artifact.pathname.startswith("viewable_media/match-1/put-token/")
    assert client.put_options == {
        "access": "public",
        "contentType": "video/mp4",
        "addRandomSuffix": True,
        "overwrite": False,
        "multipart": False,
    }
    assert "server-only-secret" not in repr(artifact.to_document())
    assert asyncio.run(store.exists(artifact)) is True
    destination = asyncio.run(store.get(artifact, tmp_path / "downloaded.mp4"))
    assert destination.read_bytes() == b"viewable-media"
    asyncio.run(store.delete(artifact))
    assert asyncio.run(store.exists(artifact)) is False


def test_source_and_internal_artifacts_cannot_be_public(tmp_path: Path) -> None:
    with pytest.raises(PersistenceValidationError):
        ArtifactPutRequest(
            source_path=tmp_path / "source.mp4",
            artifact_type="source_video",
            category=ArtifactCategory.SOURCE_MEDIA,
            access=ArtifactAccess.PUBLIC,
        )


def test_local_factory_requires_no_hosted_credentials(tmp_path: Path) -> None:
    store = create_artifact_store(PersistenceSettings(local_artifact_root=tmp_path))

    assert isinstance(store, LocalArtifactStore)
