"""Overlay swing trajectory on the original video and export trajectory JSON.

Output:
- <output_dir>/trajectory_overlay.mp4  — video with trajectory drawn on each frame
- <output_dir>/trajectory.json         — structured trajectory data (re-usable)

Drawing conventions:
- Observed points: solid green dot
- Repaired/interpolated points: yellow dot (so viewer knows they're estimated)
- Extrapolated racket-head points: orange dot
- Trailing tail (last N frames): fading line to show direction
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.models import Trajectory, TrajectoryPoint, VideoMeta

_TAIL_LENGTH = 15  # number of trailing frames to draw as fading line
_DOT_RADIUS = 5
_LINE_THICKNESS = 2

_COLOR_OBSERVED = (0, 220, 0)       # green
_COLOR_REPAIRED = (0, 220, 220)     # yellow-ish (BGR)
_COLOR_EXTRAPOLATED = (0, 140, 255) # orange
_COLOR_TAIL = (180, 180, 180)       # grey


def overlay_trajectory(
    trajectory: Trajectory,
    output_dir: Path,
    also_draw_skeleton: bool = False,
) -> Path:
    """Render trajectory overlay video. Returns path to output video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = trajectory.meta
    points = trajectory.points

    frame_paths = sorted(meta.frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {meta.frames_dir}")

    out_path = output_dir / "trajectory_overlay.mp4"
    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, meta.fps_real, (w, h))

    pt_map = {p.frame_idx: p for p in points}

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Draw tail
        tail_start = max(0, idx - _TAIL_LENGTH)
        tail_pts = [
            pt_map[i] for i in range(tail_start, idx)
            if i in pt_map and not (np.isnan(pt_map[i].x) or np.isnan(pt_map[i].y))
        ]
        _draw_tail(frame, tail_pts)

        # Draw current point
        if idx in pt_map:
            pt = pt_map[idx]
            if not (np.isnan(pt.x) or np.isnan(pt.y)):
                _draw_point(frame, pt)

        writer.write(frame)

    writer.release()
    return out_path


def write_trajectory_json(trajectory: Trajectory, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "trajectory.json"

    payload = {
        "metadata": {
            "fps_real": trajectory.meta.fps_real,
            "is_vfr": trajectory.meta.is_vfr,
            "width": trajectory.meta.width,
            "height": trajectory.meta.height,
            "total_frames": trajectory.meta.total_frames,
            "duration": trajectory.meta.duration,
            "stroke_type": trajectory.meta.stroke_type.value,
            "dominant_hand": trajectory.meta.dominant_hand.value,
            "two_handed_backhand": trajectory.meta.two_handed_backhand,
            "source_hash": trajectory.meta.source_hash,
            "speed_note": (
                "speed_ms and speed_kmh are None unless a scale_px_per_metre "
                "reference was provided. All speeds are 2D projections and may "
                "underestimate true 3D speed when camera angle is oblique."
            ),
        },
        "trajectory": [_serialise_point(p) for p in trajectory.points],
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path


def _draw_tail(frame: np.ndarray, tail: list[TrajectoryPoint]) -> None:
    if len(tail) < 2:
        return
    for i in range(1, len(tail)):
        alpha = i / len(tail)
        color = tuple(int(c * alpha) for c in _COLOR_TAIL)
        p1 = (int(tail[i - 1].x), int(tail[i - 1].y))
        p2 = (int(tail[i].x), int(tail[i].y))
        cv2.line(frame, p1, p2, color, _LINE_THICKNESS)


def _draw_point(frame: np.ndarray, pt: TrajectoryPoint) -> None:
    if pt.is_extrapolated:
        color = _COLOR_EXTRAPOLATED
    elif pt.is_repaired:
        color = _COLOR_REPAIRED
    else:
        color = _COLOR_OBSERVED
    center = (int(round(pt.x)), int(round(pt.y)))
    cv2.circle(frame, center, _DOT_RADIUS, color, -1)


def _serialise_point(p: TrajectoryPoint) -> dict:
    return {
        "frame_idx": p.frame_idx,
        "timestamp_sec": p.timestamp_sec,
        "x": None if np.isnan(p.x) else round(p.x, 2),
        "y": None if np.isnan(p.y) else round(p.y, 2),
        "point_type": p.point_type,
        "is_observed": p.is_observed,
        "is_repaired": p.is_repaired,
        "is_extrapolated": p.is_extrapolated,
        "speed_px_per_sec": p.speed_px_per_sec,
        "speed_ms": p.speed_ms,
        "speed_kmh": p.speed_kmh,
    }
