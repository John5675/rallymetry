from pathlib import Path

from pickleball_vision.config import PlayerTrackingSettings
from pickleball_vision.court import CourtPoint, ImagePoint
from pickleball_vision.person_detection import BoundingBox
from pickleball_vision.player_isolation import (
    LOGICAL_PLAYER_ROLES,
    CourtRegionState,
    CourtSide,
    GroundContactAssessment,
    GroundProjectionStatus,
    LogicalPlayerAssignments,
    LogicalPlayerRole,
    ManualPlayerAssignment,
)
from pickleball_vision.player_tracking import (
    LogicalTrackingState,
    RawTrackerObservation,
    TrackerMetadata,
    build_tracking_run,
    resolve_logical_player_tracks,
    tracking_summary,
)
from pickleball_vision.video import VideoMetadata


def _source(frame_count: int = 7) -> VideoMetadata:
    return VideoMetadata(
        "test.mp4", Path("/tmp/test.mp4"), 1920, 1080, 10.0, frame_count, 0.7, "h264"
    )


def _ground(
    x_m: float,
    y_m: float,
    *,
    region: CourtRegionState = CourtRegionState.INSIDE,
) -> GroundContactAssessment:
    return GroundContactAssessment(
        image_point=ImagePoint(100 + x_m * 10, 900 - y_m * 10),
        court_point=CourtPoint(x_m, y_m),
        projection_status=GroundProjectionStatus.PROJECTED,
        region_state=region,
        region_confidence=1.0,
        region_boundary_ambiguous=False,
        side=CourtSide.NEAR if y_m < 6.7056 else CourtSide.FAR,
        side_confidence=1.0,
    )


def _raw(frame: int, role_index: int, tracker_id: int, raw_index: int) -> RawTrackerObservation:
    return RawTrackerObservation(
        observation_id=f"o-{frame}-{role_index}-{raw_index}",
        tracker_id=tracker_id,
        raw_detection_index=raw_index,
        frame_number=frame,
        timestamp_s=frame / 10,
        tracker_bounding_box=BoundingBox(10 + role_index * 20, 10, 20 + role_index * 20, 40),
        detection_confidence=0.9,
        tracker_confidence=0.9,
    )


def _assignments(
    anchor_indices: list[int],
    *,
    anchor_frame: int = 2,
) -> LogicalPlayerAssignments:
    return LogicalPlayerAssignments(
        created_at_utc="2026-08-14T00:00:00+00:00",
        candidates_path="/tmp/candidates.json",
        detections_path="/tmp/detections.json",
        assignments=tuple(
            ManualPlayerAssignment(
                role,
                f"candidate-{index}",
                anchor_indices[index],
                anchor_frame,
                anchor_frame / 10,
                CourtSide.NEAR if index < 2 else CourtSide.FAR,
            )
            for index, role in enumerate(LOGICAL_PLAYER_ROLES)
        ),
    )


def test_logical_identity_survives_missed_frame_and_tracker_id_change() -> None:
    positions = ((1.0, 2.0), (5.0, 2.0), (1.0, 14.5), (5.0, 11.0))
    observations: list[RawTrackerObservation] = []
    grounds: dict[str, GroundContactAssessment] = {}
    anchor_indices: list[int] = []
    next_raw_index = 0
    for frame in range(7):
        for role_index, (x_m, y_m) in enumerate(positions):
            if frame == 3 and role_index == 0:
                continue
            tracker_id = 99 if role_index == 0 and frame >= 4 else role_index + 1
            item = _raw(frame, role_index, tracker_id, next_raw_index)
            observations.append(item)
            grounds[item.observation_id] = _ground(
                x_m + frame * 0.02,
                y_m,
                region=(CourtRegionState.OUTSIDE if role_index == 2 else CourtRegionState.INSIDE),
            )
            if frame == 2:
                anchor_indices.append(next_raw_index)
            next_raw_index += 1
        if frame == 4:
            distractor = _raw(frame, 8, 200, next_raw_index)
            observations.append(distractor)
            grounds[distractor.observation_id] = _ground(8.5, 2.0, region=CourtRegionState.OUTSIDE)
            next_raw_index += 1

    tracks, switches = resolve_logical_player_tracks(
        source=_source(),
        raw_observations=tuple(observations),
        ground_by_observation_id=grounds,
        assignments=_assignments(anchor_indices),
        settings=PlayerTrackingSettings(),
    )

    me = tracks[LOGICAL_PLAYER_ROLES[0]]
    assert me[3].state is LogicalTrackingState.TEMPORARILY_MISSING
    assert me[4].state is LogicalTrackingState.REACQUIRED
    assert me[4].tracker_observation is not None
    assert me[4].tracker_observation.raw.tracker_id == 99
    assert not switches
    assert all(
        frame.tracker_observation is not None
        for role in LOGICAL_PLAYER_ROLES[1:]
        for frame in tracks[role]
    )


def test_immediate_tracker_id_change_is_surfaced_for_review() -> None:
    positions = ((1.0, 2.0), (5.0, 2.0), (1.0, 11.0), (5.0, 11.0))
    observations: list[RawTrackerObservation] = []
    grounds: dict[str, GroundContactAssessment] = {}
    anchor_indices: list[int] = []
    raw_index = 0
    for frame in range(5):
        for role_index, (x_m, y_m) in enumerate(positions):
            tracker_id = 77 if role_index == 0 and frame >= 3 else role_index + 1
            item = _raw(frame, role_index, tracker_id, raw_index)
            observations.append(item)
            grounds[item.observation_id] = _ground(x_m, y_m)
            if frame == 2:
                anchor_indices.append(raw_index)
            raw_index += 1

    tracks, switches = resolve_logical_player_tracks(
        source=_source(frame_count=5),
        raw_observations=tuple(observations),
        ground_by_observation_id=grounds,
        assignments=_assignments(anchor_indices),
        settings=PlayerTrackingSettings(),
    )

    assert (
        tracks[LOGICAL_PLAYER_ROLES[0]][3].state is LogicalTrackingState.SUSPECTED_IDENTITY_SWITCH
    )
    assert len(switches) == 1
    assert switches[0].previous_tracker_id == 1
    assert switches[0].current_tracker_id == 77


def test_tracking_summary_reports_coverage_missing_and_reacquisition() -> None:
    source = _source()
    positions = ((1.0, 2.0), (5.0, 2.0), (1.0, 11.0), (5.0, 11.0))
    observations: list[RawTrackerObservation] = []
    grounds: dict[str, GroundContactAssessment] = {}
    anchor_indices: list[int] = []
    raw_index = 0
    for frame in range(source.frame_count):
        for role_index, (x_m, y_m) in enumerate(positions):
            if role_index == 0 and frame == 3:
                continue
            item = _raw(frame, role_index, role_index + 1, raw_index)
            observations.append(item)
            grounds[item.observation_id] = _ground(x_m, y_m)
            if frame == 2:
                anchor_indices.append(raw_index)
            raw_index += 1
    tracks, switches = resolve_logical_player_tracks(
        source=source,
        raw_observations=tuple(observations),
        ground_by_observation_id=grounds,
        assignments=_assignments(anchor_indices),
        settings=PlayerTrackingSettings(),
    )
    run = build_tracking_run(
        source=source,
        detections_path="d.json",
        candidates_path="candidates.json",
        assignments_path="a.json",
        calibration_path="c.json",
        tracker=TrackerMetadata("test", "test", "test", "1", {}),
        configuration={},
        appearance={},
        player_names={role: role.value for role in LOGICAL_PLAYER_ROLES},
        raw_observations=tuple(observations),
        logical_tracks=tracks,
        suspected_identity_switches=switches,
    )

    summary = tracking_summary(run, artifacts={})
    me = summary["players"]["ME"]  # type: ignore[index]
    assert me["observed_frames"] == 6
    assert me["reacquisition_count"] == 1
    assert me["longest_missing_interval"]["frames"] == 1


def test_appearance_keeps_opponents_correct_when_tracker_ids_swap() -> None:
    source = _source(frame_count=5)
    x_positions = (
        (2.0, 4.0),
        (2.5, 3.5),
        (2.9, 3.1),
        (3.3, 2.7),
        (3.7, 2.3),
    )
    observations: list[RawTrackerObservation] = []
    grounds: dict[str, GroundContactAssessment] = {}
    similarities: dict[LogicalPlayerRole, dict[str, float]] = {
        role: {} for role in LOGICAL_PLAYER_ROLES
    }
    anchor_indices: list[int] = []
    raw_index = 0
    for frame_number in range(source.frame_count):
        physical_positions = (
            (1.0, 2.0),
            (5.0, 2.0),
            (x_positions[frame_number][0], 11.0),
            (x_positions[frame_number][1], 11.0),
        )
        for physical_role_index, (x_m, y_m) in enumerate(physical_positions):
            tracker_id = physical_role_index + 1
            if frame_number >= 3 and physical_role_index >= 2:
                tracker_id = 7 - tracker_id
            item = _raw(frame_number, physical_role_index, tracker_id, raw_index)
            observations.append(item)
            grounds[item.observation_id] = _ground(x_m, y_m)
            physical_role = LOGICAL_PLAYER_ROLES[physical_role_index]
            same_side_competitor = (
                LOGICAL_PLAYER_ROLES[1 - physical_role_index]
                if physical_role_index < 2
                else LOGICAL_PLAYER_ROLES[5 - physical_role_index]
            )
            similarities[physical_role][item.observation_id] = 0.95
            similarities[same_side_competitor][item.observation_id] = 0.50
            if frame_number == 2:
                anchor_indices.append(raw_index)
            raw_index += 1

    tracks, switches = resolve_logical_player_tracks(
        source=source,
        raw_observations=tuple(observations),
        ground_by_observation_id=grounds,
        assignments=_assignments(anchor_indices),
        settings=PlayerTrackingSettings(),
        appearance_similarities=similarities,
    )

    oksana = tracks[LOGICAL_PLAYER_ROLES[2]]
    diana = tracks[LOGICAL_PLAYER_ROLES[3]]
    assert oksana[3].tracker_observation is not None
    assert diana[3].tracker_observation is not None
    assert oksana[3].tracker_observation.raw.raw_detection_index == 14
    assert diana[3].tracker_observation.raw.raw_detection_index == 15
    assert oksana[3].tracker_observation.raw.tracker_id == 4
    assert diana[3].tracker_observation.raw.tracker_id == 3
    assert {event.logical_player for event in switches} >= {
        LOGICAL_PLAYER_ROLES[2],
        LOGICAL_PLAYER_ROLES[3],
    }


def test_strong_appearance_reacquires_player_across_long_gap() -> None:
    source = _source(frame_count=50)
    positions = ((1.0, 2.0), (5.0, 2.0), (1.0, 11.0), (5.0, 11.0))
    observations: list[RawTrackerObservation] = []
    grounds: dict[str, GroundContactAssessment] = {}
    similarities: dict[LogicalPlayerRole, dict[str, float]] = {
        role: {} for role in LOGICAL_PLAYER_ROLES
    }
    anchor_indices: list[int] = []
    raw_index = 0
    for frame_number in range(source.frame_count):
        for role_index, (x_m, y_m) in enumerate(positions):
            if role_index == 0 and 6 <= frame_number < 46:
                continue
            tracker_id = role_index + 1
            if role_index == 0 and frame_number >= 46:
                tracker_id = 99
            item = _raw(frame_number, role_index, tracker_id, raw_index)
            observations.append(item)
            grounds[item.observation_id] = _ground(x_m, y_m)
            own_role = LOGICAL_PLAYER_ROLES[role_index]
            same_side_competitor = (
                LOGICAL_PLAYER_ROLES[1 - role_index]
                if role_index < 2
                else LOGICAL_PLAYER_ROLES[5 - role_index]
            )
            similarities[own_role][item.observation_id] = 0.95
            similarities[same_side_competitor][item.observation_id] = 0.50
            if frame_number == 46:
                anchor_indices.append(raw_index)
            raw_index += 1

    tracks, _ = resolve_logical_player_tracks(
        source=source,
        raw_observations=tuple(observations),
        ground_by_observation_id=grounds,
        assignments=_assignments(anchor_indices, anchor_frame=46),
        settings=PlayerTrackingSettings(),
        appearance_similarities=similarities,
    )

    me = tracks[LOGICAL_PLAYER_ROLES[0]]
    assert me[5].tracker_observation is not None
    assert me[5].tracker_observation.raw.tracker_id == 1
    assert me[6].state is LogicalTrackingState.TEMPORARILY_MISSING
    assert me[46].tracker_observation is not None
    assert me[46].tracker_observation.raw.tracker_id == 99
