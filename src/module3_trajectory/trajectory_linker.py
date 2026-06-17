"""Connect per-frame wrist/racket-head candidates into a trajectory.

Strategy:
- Use the wrist_primary point when valid; fall back to racket_head_approx.
- Invalid frames are kept in the sequence with is_observed=False so gap_filler
  can see them and interpolate.
- Output is frame-contiguous (one TrajectoryPoint per input frame).
"""

from __future__ import annotations

from src.models import FrameJoints, TrajectoryPoint, VideoMeta


def link_trajectory(
    frames: list[FrameJoints],
    meta: VideoMeta,
    prefer_racket_head: bool = False,
) -> list[TrajectoryPoint]:
    """Convert per-frame FrameJoints into an ordered TrajectoryPoint sequence.

    prefer_racket_head=True uses the extrapolated racket-head point instead of
    the wrist when available (and marks is_extrapolated accordingly).
    """
    points: list[TrajectoryPoint] = []

    for fj in frames:
        pt = _pick_point(fj, prefer_racket_head)
        points.append(pt)

    return points


def _pick_point(fj: FrameJoints, prefer_racket_head: bool) -> TrajectoryPoint:
    is_extrapolated = False
    point_type = "wrist"

    src = fj.wrist_primary
    if prefer_racket_head and fj.racket_head_approx is not None:
        src = None  # use racket head
        rh = fj.racket_head_approx
        point_type = "racket_head_approx"
        is_extrapolated = rh.is_extrapolated
        x, y = rh.x, rh.y
        is_observed = fj.valid
    elif src is not None:
        x, y = src.x, src.y
        is_observed = fj.valid
    else:
        # No usable point — emit NaN placeholder so gap_filler sees the gap
        x, y = float("nan"), float("nan")
        is_observed = False

    if src is not None:
        x, y = src.x, src.y
        is_observed = fj.valid

    return TrajectoryPoint(
        frame_idx=fj.frame_idx,
        timestamp_sec=fj.timestamp_sec,
        x=x,
        y=y,
        point_type=point_type,
        is_observed=is_observed,
        is_repaired=False,
        is_extrapolated=is_extrapolated,
    )
