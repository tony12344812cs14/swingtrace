"""Shared data models for the tennis swing analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class StrokeType(str, Enum):
    FOREHAND = "forehand"
    BACKHAND_ONE = "backhand_one"
    BACKHAND_TWO = "backhand_two"
    SERVE = "serve"
    VOLLEY = "volley"
    UNKNOWN = "unknown"


class DominantHand(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class QualityChecks:
    fps_ok: bool
    fps_warning: Optional[str]
    stability_ok: bool
    stability_score: float          # mean inter-frame displacement in pixels
    player_visibility: str          # "ok" | "warning" | "fail"
    player_visibility_note: Optional[str] = None


@dataclass
class VideoMeta:
    fps_real: float
    is_vfr: bool
    frame_timestamps: list[float]   # per-frame timestamps (always present)
    width: int
    height: int
    total_frames: int
    duration: float
    stroke_type: StrokeType
    dominant_hand: DominantHand
    two_handed_backhand: bool
    checks: QualityChecks
    source_hash: str                # SHA-256 of source file
    frames_dir: Path                # directory containing extracted frames


@dataclass
class JointCoord:
    x: float                        # normalised [0,1] or pixel depending on context
    y: float
    z: float
    confidence: float


@dataclass
class RacketHeadApprox:
    x: float
    y: float
    is_extrapolated: bool = True


@dataclass
class FrameJoints:
    frame_idx: int
    timestamp_sec: float
    valid: bool                     # False = low confidence / occluded
    joints: dict[str, JointCoord]   # key: joint name (e.g. "right_wrist")
    wrist_primary: Optional[JointCoord] = None   # resolved primary wrist point
    racket_head_approx: Optional[RacketHeadApprox] = None


@dataclass
class TrajectoryPoint:
    frame_idx: int
    timestamp_sec: float
    x: float                        # pixel coordinate
    y: float
    point_type: str                 # "wrist" | "racket_head_approx"
    is_observed: bool
    is_repaired: bool
    is_extrapolated: bool
    speed_px_per_sec: Optional[float] = None
    speed_ms: Optional[float] = None    # None if no scale reference
    speed_kmh: Optional[float] = None


@dataclass
class Trajectory:
    meta: VideoMeta
    points: list[TrajectoryPoint] = field(default_factory=list)
