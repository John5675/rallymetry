"""Strict YouTube URL parsing for user-submitted match recordings."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from pickleball_vision.api.errors import ApiError

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_STANDARD_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }
)
_SHORT_HOSTS = frozenset({"youtu.be", "www.youtu.be"})
_PATH_PREFIXES = frozenset({"embed", "live", "shorts"})


def parse_youtube_video_id(value: str) -> str:
    """Return one canonical video ID without following redirects or fetching a URL."""

    raw = value.strip()
    if not raw or len(raw) > 2048:
        raise _invalid_url()
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or parsed.username or parsed.password:
        raise _invalid_url()
    host = (parsed.hostname or "").lower().rstrip(".")
    candidate: str | None = None
    path_parts = [part for part in parsed.path.split("/") if part]
    if host in _SHORT_HOSTS and path_parts:
        candidate = path_parts[0]
    elif host in _STANDARD_HOSTS:
        if parsed.path.rstrip("/") == "/watch":
            values = parse_qs(parsed.query).get("v", [])
            candidate = values[0] if values else None
        elif len(path_parts) == 2 and path_parts[0] in _PATH_PREFIXES:
            candidate = path_parts[1]
    if candidate is None or _VIDEO_ID.fullmatch(candidate) is None:
        raise _invalid_url()
    return candidate


def _invalid_url() -> ApiError:
    return ApiError(
        status_code=422,
        code="youtube_url_invalid",
        message="Enter a valid YouTube video link, not a channel or playlist link",
    )
