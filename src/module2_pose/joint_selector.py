"""Resolve which joints to use based on dominant_hand and stroke_type.

Rules (per SKILL.md):
- Core: dominant-hand shoulder/elbow/wrist
- Two-handed backhand + stroke_type==backhand_two: take BOTH wrists (midpoint as primary)
- Serve: allow low-confidence frames in trophy/layback phase (normal self-occlusion)
- dominant_hand determines left vs right; never mix them up
"""

from __future__ import annotations

import math
from typing import Optional

from src.models import DominantHand, JointCoord, RacketHeadApprox, StrokeType

# Racket approximate length in pixels relative to forearm length (heuristic)
_RACKET_APPROX_FOREARM_MULTIPLIER = 1.8


def select_primary_wrist(
    joints: dict[str, JointCoord],
    dominant_hand: DominantHand,
    stroke_type: StrokeType,
    two_handed_backhand: bool,
) -> Optional[JointCoord]:
    """Return the primary wrist coordinate to use as trajectory anchor.

    For two-handed backhand: returns midpoint of both wrists.
    Otherwise: dominant-hand wrist.
    """
    is_two_hand_backhand = (
        two_handed_backhand and stroke_type == StrokeType.BACKHAND_TWO
    )

    if is_two_hand_backhand:
        return _midpoint_wrist(joints)

    hand = dominant_hand.value  # "left" or "right"
    return joints.get(f"{hand}_wrist")


def _midpoint_wrist(joints: dict[str, JointCoord]) -> Optional[JointCoord]:
    lw = joints.get("left_wrist")
    rw = joints.get("right_wrist")
    if lw is None or rw is None:
        return lw or rw
    return JointCoord(
        x=(lw.x + rw.x) / 2,
        y=(lw.y + rw.y) / 2,
        z=(lw.z + rw.z) / 2,
        confidence=min(lw.confidence, rw.confidence),
    )


def extrapolate_racket_head(
    wrist: JointCoord,
    elbow: JointCoord,
) -> RacketHeadApprox:
    """Approximate racket-head position by extending the forearm vector.

    This is a simple geometric estimate — NOT a real racket detector.
    Always marked is_extrapolated=True so consumers know it's an approximation.
    """
    dx = wrist.x - elbow.x
    dy = wrist.y - elbow.y
    length = math.hypot(dx, dy)

    if length < 1e-6:
        return RacketHeadApprox(x=wrist.x, y=wrist.y, is_extrapolated=True)

    nx, ny = dx / length, dy / length
    scale = length * _RACKET_APPROX_FOREARM_MULTIPLIER

    return RacketHeadApprox(
        x=wrist.x + nx * scale,
        y=wrist.y + ny * scale,
        is_extrapolated=True,
    )
