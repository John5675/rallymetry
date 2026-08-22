from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar, Self

import pytest

import pickleball_vision.analysis_runtime.youtube as youtube_module
from pickleball_vision.analysis_runtime.youtube import YtDlpYouTubeDownloader
from pickleball_vision.errors import AnalysisSourceError


class FakeYoutubeDL:
    duration_seconds = 60.0
    options_seen: ClassVar[list[dict[str, object]]] = []

    def __init__(self, options: dict[str, object]) -> None:
        self.options = options
        self.options_seen.append(options)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def extract_info(self, url: str, *, download: bool) -> dict[str, object]:
        assert url == "https://www.youtube.com/watch?v=_cPF1fTnk0Y"
        if download:
            template = self.options["outtmpl"]
            assert isinstance(template, str)
            destination = Path(template.replace("%(ext)s", "mp4"))
            destination.write_bytes(b"synthetic youtube media")
        return {"duration": self.duration_seconds}


def test_youtube_downloader_preserves_one_media_file_with_bounded_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeYoutubeDL.duration_seconds = 60.0
    FakeYoutubeDL.options_seen = []
    monkeypatch.setattr(youtube_module, "YoutubeDL", FakeYoutubeDL)
    downloader = YtDlpYouTubeDownloader(max_duration_seconds=120, max_bytes=10_000)

    result = asyncio.run(downloader.download("_cPF1fTnk0Y", destination_dir=tmp_path / "input"))

    assert result == (tmp_path / "input" / "source.mp4").resolve()
    assert result.read_bytes() == b"synthetic youtube media"
    download_options = FakeYoutubeDL.options_seen[-1]
    assert download_options["noplaylist"] is True
    assert download_options["max_filesize"] == 10_000
    assert "bestaudio" in str(download_options["format"])


def test_youtube_downloader_configures_server_side_challenge_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeYoutubeDL.duration_seconds = 60.0
    FakeYoutubeDL.options_seen = []
    monkeypatch.setattr(youtube_module, "YoutubeDL", FakeYoutubeDL)
    downloader = YtDlpYouTubeDownloader(
        max_duration_seconds=120,
        max_bytes=10_000,
        pot_provider_url="http://youtube-pot-provider:4416",
    )

    asyncio.run(downloader.download("_cPF1fTnk0Y", destination_dir=tmp_path / "input"))

    for options in FakeYoutubeDL.options_seen:
        assert options["js_runtimes"] == {"node": {}}
        extractor_args = options["extractor_args"]
        assert isinstance(extractor_args, dict)
        assert extractor_args["youtube"] == {
            "player_client": ["web_embedded", "mweb"],
            "player_skip": ["webpage"],
        }
        assert extractor_args["youtubepot-bgutilhttp"] == {
            "base_url": ["http://youtube-pot-provider:4416"],
        }


def test_youtube_downloader_prefers_configured_proxy_over_challenge_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeYoutubeDL.duration_seconds = 60.0
    FakeYoutubeDL.options_seen = []
    monkeypatch.setattr(youtube_module, "YoutubeDL", FakeYoutubeDL)
    downloader = YtDlpYouTubeDownloader(
        max_duration_seconds=120,
        max_bytes=10_000,
        pot_provider_url="http://youtube-pot-provider:4416",
        proxy_url="http://user:password@residential-proxy.example:1234",
    )

    asyncio.run(downloader.download("_cPF1fTnk0Y", destination_dir=tmp_path / "input"))

    for options in FakeYoutubeDL.options_seen:
        assert options["js_runtimes"] == {"node": {}}
        assert options["proxy"] == "http://user:password@residential-proxy.example:1234"
        assert "extractor_args" not in options


def test_youtube_downloader_rejects_overlong_media_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeYoutubeDL.duration_seconds = 121.0
    FakeYoutubeDL.options_seen = []
    monkeypatch.setattr(youtube_module, "YoutubeDL", FakeYoutubeDL)
    downloader = YtDlpYouTubeDownloader(max_duration_seconds=120, max_bytes=10_000)

    with pytest.raises(AnalysisSourceError, match="duration limit"):
        asyncio.run(downloader.download("_cPF1fTnk0Y", destination_dir=tmp_path / "input"))

    assert len(FakeYoutubeDL.options_seen) == 1
    assert not tuple((tmp_path / "input").glob("source.*"))
