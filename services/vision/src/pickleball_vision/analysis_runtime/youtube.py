"""Bounded YouTube source retrieval for temporary workflow workspaces only."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

import imageio_ffmpeg  # type: ignore[import-untyped]
from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from pickleball_vision.errors import AnalysisSourceError


class YouTubeDownloader(Protocol):
    async def download(self, video_id: str, *, destination_dir: Path) -> Path: ...


class YtDlpYouTubeDownloader:
    """Download one accessible YouTube video with audio and bounded resource use."""

    def __init__(self, *, max_duration_seconds: int, max_bytes: int) -> None:
        self._max_duration_seconds = max_duration_seconds
        self._max_bytes = max_bytes

    async def download(self, video_id: str, *, destination_dir: Path) -> Path:
        return await asyncio.to_thread(self._download, video_id, destination_dir)

    def _download(self, video_id: str, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://www.youtube.com/watch?v={video_id}"
        common_options: dict[str, object] = {
            "cachedir": False,
            "extract_flat": False,
            "noplaylist": True,
            "no_warnings": True,
            "quiet": True,
            "socket_timeout": 30,
        }
        try:
            with YoutubeDL({**common_options, "skip_download": True}) as inspector:
                metadata = inspector.extract_info(url, download=False)
            if not isinstance(metadata, dict):
                raise AnalysisSourceError("YouTube returned invalid video metadata")
            duration = metadata.get("duration")
            if (
                isinstance(duration, (int, float))
                and not isinstance(duration, bool)
                and duration > self._max_duration_seconds
            ):
                raise AnalysisSourceError(
                    "YouTube recording exceeds the configured analysis duration limit"
                )
            options = {
                **common_options,
                "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
                "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
                "max_filesize": self._max_bytes,
                "merge_output_format": "mp4",
                "outtmpl": str(destination_dir / "source.%(ext)s"),
                "overwrites": False,
            }
            with YoutubeDL(options) as downloader:
                downloader.extract_info(url, download=True)
        except AnalysisSourceError:
            raise
        except DownloadError as error:
            raise AnalysisSourceError(
                "YouTube recording could not be downloaded; verify it is accessible"
            ) from error
        except OSError as error:
            raise AnalysisSourceError("YouTube source could not be written") from error

        candidates = tuple(
            path
            for path in destination_dir.glob("source.*")
            if path.is_file() and path.suffix not in {".part", ".ytdl"}
        )
        if not candidates:
            raise AnalysisSourceError("YouTube download completed without a media file")
        result = max(candidates, key=lambda path: path.stat().st_size)
        if result.stat().st_size > self._max_bytes:
            raise AnalysisSourceError("YouTube recording exceeds the configured size limit")
        return result.resolve()
