"""Bidirectional gap filling for occluded / low-confidence frames.

Offline analysis means we have ALL frames available — use both forward and
backward context to repair missing points (cubic interpolation where gap is
short; linear for longer gaps).

Key principle: repair gaps but MARK them as is_repaired=True so the caller
knows which points are real observations vs reconstructions.
"""

from __future__ import annotations

import math

import numpy as np

from src.models import TrajectoryPoint

_SHORT_GAP_THRESHOLD = 5  # frames; use cubic spline for gaps <= this
_MAX_REPAIRABLE_GAP = 30  # longer gaps are left as NaN (too uncertain)


def fill_gaps(points: list[TrajectoryPoint]) -> list[TrajectoryPoint]:
    """Fill NaN or invalid entries using bidirectional interpolation.

    Returns a NEW list; original objects are not mutated.
    """
    pts = [_clone(p) for p in points]

    xs = np.array([p.x for p in pts], dtype=float)
    ys = np.array([p.y for p in pts], dtype=float)

    valid_mask = np.array([p.is_observed for p in pts], dtype=bool)
    nan_mask = np.isnan(xs) | np.isnan(ys)
    needs_repair = (~valid_mask) | nan_mask

    gap_groups = _find_gap_groups(needs_repair)

    for start, end in gap_groups:
        gap_len = end - start + 1
        if gap_len > _MAX_REPAIRABLE_GAP:
            continue  # leave unrepaired

        # Find anchor frames on both sides
        left_idx = start - 1
        right_idx = end + 1

        if left_idx < 0 and right_idx >= len(pts):
            continue  # no anchors at all
        if left_idx < 0:
            # Extrapolate forward from right anchor only (not ideal, but better than NaN)
            for i in range(start, end + 1):
                xs[i] = xs[right_idx]
                ys[i] = ys[right_idx]
        elif right_idx >= len(pts):
            # Extrapolate backward from left anchor
            for i in range(start, end + 1):
                xs[i] = xs[left_idx]
                ys[i] = ys[left_idx]
        else:
            # Bidirectional: interpolate between left and right anchors
            _interpolate_segment(xs, ys, left_idx, right_idx, gap_len)

        for i in range(start, end + 1):
            pts[i].x = float(xs[i])
            pts[i].y = float(ys[i])
            pts[i].is_repaired = True
            pts[i].is_observed = False

    return pts


def _interpolate_segment(
    xs: np.ndarray,
    ys: np.ndarray,
    left: int,
    right: int,
    gap_len: int,
) -> None:
    indices = np.arange(left, right + 1)
    t = np.array([left, right], dtype=float)
    known_x = np.array([xs[left], xs[right]])
    known_y = np.array([ys[left], ys[right]])

    if gap_len <= _SHORT_GAP_THRESHOLD:
        # Cubic Hermite via numpy interp (linear for 2-point; upgrade to scipy if needed)
        try:
            from scipy.interpolate import CubicSpline
            cs_x = CubicSpline(t, known_x)
            cs_y = CubicSpline(t, known_y)
            xs[left + 1:right] = cs_x(indices[1:-1])
            ys[left + 1:right] = cs_y(indices[1:-1])
            return
        except ImportError:
            pass

    # Linear fallback
    xs[left + 1:right] = np.interp(indices[1:-1], t, known_x)
    ys[left + 1:right] = np.interp(indices[1:-1], t, known_y)


def _find_gap_groups(needs_repair: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index ranges for consecutive True entries."""
    groups: list[tuple[int, int]] = []
    n = len(needs_repair)
    i = 0
    while i < n:
        if needs_repair[i]:
            j = i
            while j < n and needs_repair[j]:
                j += 1
            groups.append((i, j - 1))
            i = j
        else:
            i += 1
    return groups


def _clone(p: TrajectoryPoint) -> TrajectoryPoint:
    return TrajectoryPoint(
        frame_idx=p.frame_idx,
        timestamp_sec=p.timestamp_sec,
        x=p.x,
        y=p.y,
        point_type=p.point_type,
        is_observed=p.is_observed,
        is_repaired=p.is_repaired,
        is_extrapolated=p.is_extrapolated,
        speed_px_per_sec=p.speed_px_per_sec,
        speed_ms=p.speed_ms,
        speed_kmh=p.speed_kmh,
    )
