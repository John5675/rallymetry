"""Model adapters for stable project-owned detection protocols."""

from pickleball_vision.detectors.ultralytics_ball import UltralyticsBallDetector
from pickleball_vision.detectors.ultralytics_person import UltralyticsPersonDetector

__all__ = ["UltralyticsBallDetector", "UltralyticsPersonDetector"]
