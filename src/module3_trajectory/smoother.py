"""Smooth the trajectory while preserving contact-point inflections.

Key constraint from SKILL.md:
- Swing trajectory follows human-arm kinematics (smooth arc), NOT ballistic physics.
- The contact instant is a sharp direction/speed change — DO NOT over-smooth it.
- Use data-driven smoothing (Savitzky-Golay) rather than any physics model.

Strategy:
1. Detect inflection candidates (local speed peak + direction change).
2. Smooth the trajectory globally with Savitzky-Golay.
3. Restore original values around inflection frames (±window).
"""

from __future__ import annotations

import numpy as np

from src.models import TrajectoryPoint

_SAVGOL_WINDOW = 11   # must be odd
_SAVGOL_POLY = 3
_INFLECTION_PROTECT_HALF = 2  # frames on each side of inflection to preserve


def smooth_trajectory(
    points: list[TrajectoryPoint],
    window: int = _SAVGOL_WINDOW,
) -> list[TrajectoryPoint]:
    """Return new trajectory with smoothed coords; inflection frames are restored."""
    if len(points) < window:
        return points

    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)

    inflection_frames = _detect_inflections(xs, ys)

    xs_smooth = _savgol(xs, window)
    ys_smooth = _savgol(ys, window)

    # Restore inflection neighbourhood
    protect = set()
    for fi in inflection_frames:
        for offset in range(-_INFLECTION_PROTECT_HALF, _INFLECTION_PROTECT_HALF + 1):
            idx = fi + offset
            if 0 <= idx < len(points):
                protect.add(idx)

    for i in protect:
        xs_smooth[i] = xs[i]
        ys_smooth[i] = ys[i]

    result = []
    for i, p in enumerate(points):
        from src.module3_trajectory.gap_filler import _clone
        np_ = _clone(p)
        np_.x = float(xs_smooth[i])
        np_.y = float(ys_smooth[i])
        result.append(np_)

    return result


def _savgol(arr: np.ndarray, window: int) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
        w = window if window % 2 == 1 else window + 1
        w = min(w, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
        if w < 3:
            return arr.copy()
        return savgol_filter(arr, window_length=w, polyorder=_SAVGOL_POLY)
    except ImportError:
        # Simple moving-average fallback
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")


def _detect_inflections(xs: np.ndarray, ys: np.ndarray) -> list[int]:
    """Find frames where speed peaks AND direction changes sharply (contact instant)."""
    if len(xs) < 3:
        return []

    dx = np.diff(xs)
    dy = np.diff(ys)
    speeds = np.hypot(dx, dy)

    if len(speeds) < 3:
        return []

    # Speed local maxima
    speed_peaks = []
    for i in range(1, len(speeds) - 1):
        if speeds[i] > speeds[i - 1] and speeds[i] > speeds[i + 1]:
            speed_peaks.append(i)

    # Among peaks, keep those with a sharp direction change
    inflections = []
    for i in speed_peaks:
        if i + 1 >= len(dx):
            continue
        # Dot product of consecutive velocity vectors
        v1 = np.array([dx[i - 1], dy[i - 1]])
        v2 = np.array([dx[i], dy[i]])
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        # cos < 0.5 → direction changed by >60°
        if cos_angle < 0.5:
            inflections.append(i + 1)  # +1 because diff shifted index

    return inflections
