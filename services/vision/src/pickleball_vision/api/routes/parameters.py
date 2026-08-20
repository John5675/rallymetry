"""Validated application-ID path parameters shared by route modules."""

from typing import Annotated

from fastapi import Path

MatchId = Annotated[
    str,
    Path(
        alias="matchId",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
JobId = Annotated[
    str,
    Path(
        alias="jobId",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
