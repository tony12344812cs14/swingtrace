"""Extract frames from an MP4 and record per-frame timestamps.

For VFR videos each frame gets its own timestamp; CFR uses uniform spacing.
Frames are stored as PNG files so downstream modules never touch the original MP4.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def extract_frames(
    video_path: Path,
    output_dir: Path,
    is_vfr: bool,
    fps_real: float,
) -> list[float]:
    """Extract all frames to output_dir/*.png, return per-frame timestamps (seconds).

    For VFR videos, uses ffmpeg to write timestamps to a sidecar file so that
    each frame's true time is known. For CFR, timestamps are inferred from fps_real.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_vfr:
        return _extract_vfr(video_path, output_dir)
    else:
        return _extract_cfr(video_path, output_dir, fps_real)


def _extract_cfr(video_path: Path, output_dir: Path, fps_real: float) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    timestamps: list[float] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = output_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        timestamps.append(frame_idx / fps_real)
        frame_idx += 1

    cap.release()
    return timestamps


def _extract_vfr(video_path: Path, output_dir: Path) -> list[float]:
    """Use ffmpeg to dump frames + a pts log so each frame has its true timestamp."""
    pts_file = output_dir / "pts.txt"

    # -vf showinfo writes PTS info to stderr; we capture it
    # Use select filter to also write timestamps to a file
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"showinfo,select=1",
        "-vsync", "0",
        "-frame_pts", "1",
        str(output_dir / "%06d.png"),
        "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse pts from ffmpeg showinfo stderr output
    timestamps = _parse_showinfo_pts(result.stderr)
    if not timestamps:
        # Fallback: use cv2 with CAP_PROP_POS_MSEC
        timestamps = _extract_vfr_fallback(video_path, output_dir)

    return timestamps


def _parse_showinfo_pts(stderr: str) -> list[float]:
    """Extract pts_time values from ffmpeg showinfo filter stderr."""
    import re
    timestamps = []
    for line in stderr.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            timestamps.append(float(m.group(1)))
    return timestamps


def _extract_vfr_fallback(video_path: Path, output_dir: Path) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    timestamps: list[float] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        out_path = output_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        timestamps.append(ts)
        frame_idx += 1

    cap.release()
    return timestamps
