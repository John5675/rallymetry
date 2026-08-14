import numpy as np
import pytest

from pickleball_vision.person_detection import BoundingBox, PersonDetection
from pickleball_vision.player_appearance import (
    AppearancePrototype,
    appearance_similarity,
    extract_appearance_descriptor,
)
from pickleball_vision.player_isolation import LogicalPlayerRole


def test_two_band_histogram_distinguishes_white_and_black_clothing() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:90, 10:45] = (245, 245, 245)
    frame[10:90, 55:90] = (20, 20, 20)
    white = extract_appearance_descriptor(
        frame,
        PersonDetection(BoundingBox(10, 10, 45, 90), 0.9, 0, 0.0),
    )
    black = extract_appearance_descriptor(
        frame,
        PersonDetection(BoundingBox(55, 10, 90, 90), 0.9, 0, 0.0),
    )
    assert white is not None
    assert black is not None
    white_prototype = AppearancePrototype(
        LogicalPlayerRole.OPPONENT_1,
        white.values,
        1,
        1,
        1.0,
    )

    assert appearance_similarity(white, white_prototype) == pytest.approx(1.0)
    cross_similarity = appearance_similarity(black, white_prototype)
    assert cross_similarity is not None
    assert cross_similarity < 0.2
