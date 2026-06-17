"""Estimate swing speed from pixel displacement and fps.

Always uses ACTUAL fps from meta (VFR-aware timestamps, not assumed 30fps).

Scale reference: pixel→metre conversion requires a known-length object in frame
(racket length ≈ 0.69m, player height, shoulder width, etc.).
Without a scale reference, only pixel-per-second is reliable — never report
m/s or km/h without it (mark as None and document the limitation).

2D projection note: speed is a projected 2D value. Oblique camera angle will
underestimate the true 3D speed. Always flag this in output.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.models import TrajectoryPoint, VideoMeta


def estimate_speed(
    points: list[TrajectoryPoint],
    meta: VideoMeta,
    scale_px_per_metre: Optional[float] = None,
) -> list[TrajectoryPoint]:
    """Fill speed fields on each TrajectoryPoint.

    scale_px_per_metre: pixels per real-world metre derived from a reference
    object in frame. Pass None if unavailable (m/s and km/h stay None).
    """
    if len(points) < 2:
        return points

    timestamps = [p.timestamp_sec for p in points]
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)

    for i, pt in enumerate(points):
        if i == 0:
            dt = timestamps[1] - timestamps[0]
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
        elif i == len(points) - 1:
            dt = timestamps[-1] - timestamps[-2]
            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
        else:
            # Central difference for interior points
            dt = timestamps[i + 1] - timestamps[i - 1]
            dx = xs[i + 1] - xs[i - 1]
            dy = ys[i + 1] - ys[i - 1]

        if dt < 1e-9:
            pt.speed_px_per_sec = 0.0
            continue

        speed_px = float(np.hypot(dx, dy) / dt)
        pt.speed_px_per_sec = round(speed_px, 2)

        if scale_px_per_metre is not None and scale_px_per_metre > 0:
            speed_ms = speed_px / scale_px_per_metre
            pt.speed_ms = round(speed_ms, 3)
            pt.speed_kmh = round(speed_ms * 3.6, 2)

    return points
