"""Provider-neutral local and Vercel Blob artifact storage."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import re
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from vercel.blob import AsyncBlobClient
from vercel.blob.errors import BlobError, BlobNotFoundError

from pickleball_vision.config import ArtifactBackend, PersistenceSettings
from pickleball_vision.errors import ArtifactStorageError, PersistenceValidationError
from pickleball_vision.persistence.models import (
    ArtifactAccess,
    ArtifactCategory,
    ArtifactProvider,
    ArtifactPutRequest,
    ArtifactRecord,
)

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK_SIZE = 1024 * 1024
_MULTIPART_THRESHOLD_BYTES = 10 * 1024 * 1024


class ArtifactStore(Protocol):
    """Storage contract used by the future API and analysis worker."""

    async def put(self, request: ArtifactPutRequest) -> ArtifactRecord:
        """Persist a local file and return compact provider-neutral metadata."""

    async def get(self, artifact: ArtifactRecord, destination: Path) -> Path:
        """Materialize an artifact at a caller-selected local destination."""

    async def delete(self, artifact: ArtifactRecord) -> None:
        """Delete an artifact from this provider."""

    async def exists(self, artifact: ArtifactRecord) -> bool:
        """Return whether the provider still contains the artifact."""


class _BlobPutResult(Protocol):
    pathname: str
    url: str
    content_type: str


class _BlobHeadResult(Protocol):
    pathname: str


class _AsyncBlobClient(Protocol):
    async def upload_file(
        self,
        local_path: str | Path,
        path: str,
        *,
        access: str,
        content_type: str | None,
        add_random_suffix: bool,
        overwrite: bool,
        multipart: bool,
    ) -> _BlobPutResult: ...

    async def download_file(
        self,
        url_or_path: str,
        local_path: str | Path,
        *,
        access: str,
        overwrite: bool,
        create_parents: bool,
    ) -> str: ...

    async def delete(self, url_or_path: str) -> None: ...

    async def head(self, url_or_path: str) -> _BlobHeadResult: ...


def _random_token() -> str:
    return uuid.uuid4().hex


def _safe_filename(path: Path) -> str:
    name = _SAFE_FILENAME.sub("-", path.name).strip(".-")
    if not name:
        return "artifact.bin"
    return name[:180]


def _randomized_pathname(request: ArtifactPutRequest, token: str) -> str:
    if not token or "/" in token or ".." in token:
        raise PersistenceValidationError("artifact token factory returned an unsafe value")
    category = request.category.value.lower()
    scope = _SAFE_FILENAME.sub("-", request.match_id or "unscoped").strip(".-")
    if not scope:
        scope = "unscoped"
    return f"{category}/{scope}/{token}/{_safe_filename(request.source_path)}"


def _content_type(request: ArtifactPutRequest) -> str:
    if request.content_type is not None:
        return request.content_type
    guessed, _ = mimetypes.guess_type(request.source_path.name)
    return guessed or "application/octet-stream"


def _validate_source(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ArtifactStorageError(
            "put",
            reason="source file does not exist",
            pathname=str(resolved),
        )
    if not resolved.is_file():
        raise ArtifactStorageError(
            "put",
            reason="source path is not a file",
            pathname=str(resolved),
        )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_id(token: str) -> str:
    return f"artifact_{token}"


def _temporary_destination(destination: Path, token: str) -> Path:
    return destination.with_name(f".{destination.name}.{token}.partial")


class LocalArtifactStore:
    """Atomic filesystem-backed artifact store for offline development."""

    def __init__(
        self,
        root: Path,
        *,
        token_factory: Callable[[], str] = _random_token,
    ) -> None:
        self._root = root.expanduser().resolve()
        self._token_factory = token_factory

    @property
    def root(self) -> Path:
        return self._root

    async def put(self, request: ArtifactPutRequest) -> ArtifactRecord:
        source = _validate_source(request.source_path)
        token = self._token_factory()
        pathname = _randomized_pathname(request, token)
        target = self._resolve_path(pathname)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_destination(target, token)
        try:
            await asyncio.to_thread(shutil.copyfile, source, temporary)
            await asyncio.to_thread(os.replace, temporary, target)
            checksum = await asyncio.to_thread(_sha256, source)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ArtifactStorageError(
                "put",
                reason=str(error),
                pathname=pathname,
            ) from error
        return ArtifactRecord(
            artifact_id=_artifact_id(token),
            match_id=request.match_id,
            artifact_type=request.artifact_type,
            category=request.category,
            pathname=pathname,
            provider=ArtifactProvider.LOCAL,
            access=ArtifactAccess.LOCAL,
            content_type=_content_type(request),
            size_bytes=target.stat().st_size,
            created_at=datetime.now(UTC),
            pipeline_version=request.pipeline_version,
            checksum_sha256=checksum,
        )

    async def get(self, artifact: ArtifactRecord, destination: Path) -> Path:
        self._require_provider(artifact)
        source = self._resolve_path(artifact.pathname)
        if not source.is_file():
            raise ArtifactStorageError(
                "get",
                reason="artifact does not exist",
                pathname=artifact.pathname,
            )
        resolved_destination = destination.expanduser().resolve()
        resolved_destination.parent.mkdir(parents=True, exist_ok=True)
        token = self._token_factory()
        temporary = _temporary_destination(resolved_destination, token)
        try:
            await asyncio.to_thread(shutil.copyfile, source, temporary)
            await asyncio.to_thread(os.replace, temporary, resolved_destination)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ArtifactStorageError(
                "get",
                reason=str(error),
                pathname=artifact.pathname,
            ) from error
        return resolved_destination

    async def delete(self, artifact: ArtifactRecord) -> None:
        self._require_provider(artifact)
        target = self._resolve_path(artifact.pathname)
        try:
            await asyncio.to_thread(target.unlink, missing_ok=True)
        except OSError as error:
            raise ArtifactStorageError(
                "delete",
                reason=str(error),
                pathname=artifact.pathname,
            ) from error

    async def exists(self, artifact: ArtifactRecord) -> bool:
        self._require_provider(artifact)
        return self._resolve_path(artifact.pathname).is_file()

    def _resolve_path(self, pathname: str) -> Path:
        candidate = (self._root / pathname).resolve()
        if not candidate.is_relative_to(self._root):
            raise ArtifactStorageError(
                "resolve",
                reason="artifact pathname escapes the configured root",
                pathname=pathname,
            )
        return candidate

    @staticmethod
    def _require_provider(artifact: ArtifactRecord) -> None:
        if artifact.provider is not ArtifactProvider.LOCAL:
            raise ArtifactStorageError(
                "provider_check",
                reason="artifact belongs to a different provider",
                pathname=artifact.pathname,
            )


class VercelBlobArtifactStore:
    """Vercel Blob adapter; its token is never included in artifact metadata."""

    def __init__(
        self,
        token: str,
        *,
        client: _AsyncBlobClient | None = None,
        token_factory: Callable[[], str] = _random_token,
    ) -> None:
        if not token.strip():
            raise PersistenceValidationError("Vercel Blob token must not be empty")
        self._client = client or cast(_AsyncBlobClient, AsyncBlobClient(token=token))
        self._token_factory = token_factory

    async def put(self, request: ArtifactPutRequest) -> ArtifactRecord:
        source = _validate_source(request.source_path)
        token = self._token_factory()
        pathname = _randomized_pathname(request, token)
        access = self._access(request)
        size_bytes = source.stat().st_size
        content_type = _content_type(request)
        multipart = size_bytes >= _MULTIPART_THRESHOLD_BYTES
        try:
            result = await self._client.upload_file(
                source,
                pathname,
                access=access.value.lower(),
                content_type=content_type,
                add_random_suffix=True,
                overwrite=False,
                multipart=multipart,
            )
            checksum = await asyncio.to_thread(_sha256, source)
        except (BlobError, OSError) as error:
            raise ArtifactStorageError(
                "put",
                reason=str(error),
                pathname=pathname,
            ) from error
        return ArtifactRecord(
            artifact_id=_artifact_id(token),
            match_id=request.match_id,
            artifact_type=request.artifact_type,
            category=request.category,
            pathname=result.pathname,
            provider=ArtifactProvider.VERCEL_BLOB,
            access=access,
            content_type=result.content_type or content_type,
            size_bytes=size_bytes,
            created_at=datetime.now(UTC),
            pipeline_version=request.pipeline_version,
            url=result.url,
            checksum_sha256=checksum,
        )

    async def get(self, artifact: ArtifactRecord, destination: Path) -> Path:
        self._require_provider(artifact)
        target = artifact.url or artifact.pathname
        resolved_destination = destination.expanduser().resolve()
        resolved_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_destination(resolved_destination, self._token_factory())
        try:
            await self._client.download_file(
                target,
                temporary,
                access=artifact.access.value.lower(),
                overwrite=True,
                create_parents=True,
            )
            await asyncio.to_thread(os.replace, temporary, resolved_destination)
        except (OSError, BlobError) as error:
            temporary.unlink(missing_ok=True)
            raise ArtifactStorageError(
                "get",
                reason=str(error),
                pathname=artifact.pathname,
            ) from error
        return resolved_destination

    async def delete(self, artifact: ArtifactRecord) -> None:
        self._require_provider(artifact)
        target = artifact.url or artifact.pathname
        try:
            await self._client.delete(target)
        except BlobError as error:
            raise ArtifactStorageError(
                "delete",
                reason=str(error),
                pathname=artifact.pathname,
            ) from error

    async def exists(self, artifact: ArtifactRecord) -> bool:
        self._require_provider(artifact)
        target = artifact.url or artifact.pathname
        try:
            await self._client.head(target)
        except BlobNotFoundError:
            return False
        except BlobError as error:
            raise ArtifactStorageError(
                "exists",
                reason=str(error),
                pathname=artifact.pathname,
            ) from error
        return True

    @staticmethod
    def _access(request: ArtifactPutRequest) -> ArtifactAccess:
        access = request.access or ArtifactAccess.PRIVATE
        if access is ArtifactAccess.LOCAL:
            raise PersistenceValidationError("Vercel Blob access must be PRIVATE or PUBLIC")
        if (
            request.category is not ArtifactCategory.VIEWABLE_MEDIA
            and access is ArtifactAccess.PUBLIC
        ):
            raise PersistenceValidationError("only VIEWABLE_MEDIA may be intentionally public")
        return access

    @staticmethod
    def _require_provider(artifact: ArtifactRecord) -> None:
        if artifact.provider is not ArtifactProvider.VERCEL_BLOB:
            raise ArtifactStorageError(
                "provider_check",
                reason="artifact belongs to a different provider",
                pathname=artifact.pathname,
            )


def create_artifact_store(settings: PersistenceSettings) -> ArtifactStore:
    """Build the configured provider without exposing its credential to callers."""

    if settings.artifact_backend is ArtifactBackend.LOCAL:
        return LocalArtifactStore(settings.local_artifact_root)
    token = settings.vercel_blob_token
    if token is None:  # Defensive: PersistenceSettings.from_env already enforces this.
        raise PersistenceValidationError("Vercel Blob backend requires a configured token")
    return VercelBlobArtifactStore(token)
