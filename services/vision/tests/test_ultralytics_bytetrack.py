from pickleball_vision.config import PlayerTrackingSettings
from pickleball_vision.person_detection import BoundingBox, PersonDetection
from pickleball_vision.player_tracking import IndexedDetection
from pickleball_vision.trackers.ultralytics_bytetrack import UltralyticsByteTracker


def test_bytetrack_adapter_retains_raw_detection_indices() -> None:
    tracker = UltralyticsByteTracker(PlayerTrackingSettings(), fps=30.0)
    frames = []
    for frame_number in range(2):
        detection = PersonDetection(
            BoundingBox(10 + frame_number, 10, 30 + frame_number, 50),
            0.9,
            frame_number,
            frame_number / 30,
        )
        frames.append(
            tracker.update(
                frame_number=frame_number,
                timestamp_s=frame_number / 30,
                detections=(IndexedDetection(40 + frame_number, detection),),
                frame_width_px=100,
                frame_height_px=80,
            )
        )

    assert frames[0][0].raw_detection_index == 40
    assert frames[1][0].raw_detection_index == 41
    assert frames[0][0].tracker_id == frames[1][0].tracker_id
    assert tracker.metadata.implementation == "ByteTrack"
