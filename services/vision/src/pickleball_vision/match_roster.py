"""Conservative roster metadata parsed from recording titles.

Title metadata identifies who is listed for a match.  It does not identify which
person detection belongs to a name, nor does it establish camera-side geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DATE_PREFIX = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+")
_VERSUS = re.compile(r"\s+(?:vs\.?|versus)\s+", re.IGNORECASE)
_EXPLICIT_SEPARATOR = re.compile(r"\s*(?:&|\+|,|/)\s*")
_CAMEL_CASE_NAME = re.compile(r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?")


@dataclass(frozen=True, slots=True)
class MatchRoster:
    """Two title-declared doubles teams, without image-space identity claims."""

    raw_title: str
    teams: tuple[tuple[str, str], tuple[str, str]]

    @property
    def player_names(self) -> tuple[str, str, str, str]:
        return (*self.teams[0], *self.teams[1])

    def as_dict(self) -> dict[str, object]:
        return {
            "source": "YOUTUBE_TITLE",
            "rawTitle": self.raw_title,
            "confidence": 1.0,
            "playerNames": list(self.player_names),
            "teams": [
                {"teamId": f"TITLE_TEAM_{index}", "playerNames": list(players)}
                for index, players in enumerate(self.teams, start=1)
            ],
            "limitations": [
                "Title order does not identify image-space player detections.",
                "Title teams do not establish near-side or far-side camera geometry.",
            ],
        }


def parse_match_roster(title: str) -> MatchRoster | None:
    """Parse an unambiguous ``two players vs two players`` title.

    Supported examples include ``JohnDenny vs DianaOksana`` and
    ``John & Denny vs Diana & Oksana``.  Ambiguous titles return ``None`` so the
    application never invents a four-player roster.
    """

    cleaned = _DATE_PREFIX.sub("", title.strip(), count=1)
    sides = _VERSUS.split(cleaned, maxsplit=1)
    if len(sides) != 2:
        return None
    left = _parse_team(sides[0])
    right = _parse_team(sides[1])
    if left is None or right is None:
        return None
    names = (*left, *right)
    if len({name.casefold() for name in names}) != 4:
        return None
    return MatchRoster(raw_title=title.strip(), teams=(left, right))


def _parse_team(value: str) -> tuple[str, str] | None:
    text = value.strip()
    if not text:
        return None
    explicit = [part.strip() for part in _EXPLICIT_SEPARATOR.split(text) if part.strip()]
    if len(explicit) == 2 and all(_valid_name(part) for part in explicit):
        return explicit[0], explicit[1]
    compact = re.sub(r"\s+", "", text)
    names = _CAMEL_CASE_NAME.findall(compact)
    if len(names) == 2 and "".join(names) == compact:
        return names[0], names[1]
    return None


def _valid_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,63}", value))
