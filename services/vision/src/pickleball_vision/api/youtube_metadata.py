"""Small, optional boundary for public YouTube recording metadata."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class YouTubeTitleProvider(Protocol):
    async def title_for(self, video_id: str) -> str | None: ...


class OEmbedYouTubeTitleProvider:
    """Resolve public/unlisted video titles through YouTube's oEmbed endpoint."""

    def __init__(self, *, timeout_seconds: float = 5.0) -> None:
        self._timeout_seconds = timeout_seconds

    async def title_for(self, video_id: str) -> str | None:
        return await asyncio.to_thread(self._load_title, video_id)

    def _load_title(self, video_id: str) -> str | None:
        query = urlencode(
            {
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "format": "json",
            }
        )
        request = Request(
            f"https://www.youtube.com/oembed?{query}",
            headers={"User-Agent": "Rallymetry/0.1"},
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read(65_536).decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        title = payload.get("title")
        if not isinstance(title, str):
            return None
        normalized = title.strip()
        return normalized[:512] if normalized else None
