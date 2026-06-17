"""Read actual video metadata via ffprobe.

Never assume fps=30. Always read from the file.
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path


def read_video_meta_raw(video_path: Path) -> dict:
    """Return raw metadata dict from ffprobe.

    Raises RuntimeError if ffprobe is unavailable or the file is not a valid video.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,width,height,nb_frames,duration"
        ":format=duration,nb_streams",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found — install ffmpeg.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed: {exc.stderr.strip()}")

    data = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})

    r_frame_rate = _parse_fraction(stream.get("r_frame_rate", "0/1"))
    avg_frame_rate = _parse_fraction(stream.get("avg_frame_rate", "0/1"))

    # VFR: reported fps differs from average fps by more than 1%
    is_vfr = (
        abs(r_frame_rate - avg_frame_rate) / max(r_frame_rate, 1e-6) > 0.01
        if r_frame_rate > 0
        else False
    )

    duration_str = stream.get("duration") or fmt.get("duration", "0")
    duration = float(duration_str) if duration_str != "N/A" else 0.0

    nb_frames_str = stream.get("nb_frames", "")
    if nb_frames_str and nb_frames_str != "N/A":
        total_frames = int(nb_frames_str)
    else:
        total_frames = int(round(duration * r_frame_rate)) if r_frame_rate > 0 else 0

    return {
        "fps_real": r_frame_rate,
        "fps_avg": avg_frame_rate,
        "is_vfr": is_vfr,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "total_frames": total_frames,
        "duration": duration,
    }


def _parse_fraction(value: str) -> float:
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0
