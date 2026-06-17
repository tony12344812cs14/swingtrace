"""Quality gate for input videos.

Checks fps, camera stability, and basic player visibility.
Provides actionable warnings instead of silently producing bad results.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.models import QualityChecks

# Minimum fps recommended for fast strokes (serve / forehand).
# At 60fps a 120 km/h racket-head still moves ~8px between frames at typical scale.
_FPS_WARN_GENERAL = 30.0
_FPS_WARN_FAST_STROKE = 60.0

# Maximum mean inter-frame pixel displacement considered "stable" (tripod shot).
_STABILITY_THRESHOLD_PX = 5.0

# Minimum fraction of frames where at least one person is detected (rough check).
_PLAYER_VISIBILITY_MIN_RATIO = 0.5


def run_quality_checks(
    frames_dir: Path,
    fps_real: float,
    stroke_type: str,
    total_frames: int,
) -> QualityChecks:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        return QualityChecks(
            fps_ok=False,
            fps_warning="No frames found in output directory.",
            stability_ok=False,
            stability_score=0.0,
            player_visibility="fail",
            player_visibility_note="No frames to check.",
        )

    fps_ok, fps_warning = _check_fps(fps_real, stroke_type)
    stability_ok, stability_score = _check_stability(frame_paths)
    visibility, visibility_note = _check_player_visibility(frame_paths)

    return QualityChecks(
        fps_ok=fps_ok,
        fps_warning=fps_warning,
        stability_ok=stability_ok,
        stability_score=stability_score,
        player_visibility=visibility,
        player_visibility_note=visibility_note,
    )


def _check_fps(fps_real: float, stroke_type: str) -> tuple[bool, str | None]:
    threshold = _FPS_WARN_FAST_STROKE if stroke_type == "serve" else _FPS_WARN_GENERAL
    if fps_real < threshold:
        msg = (
            f"fps={fps_real:.1f} is below {threshold:.0f}fps recommended for "
            f"stroke_type='{stroke_type}'. Motion blur and sparse wrist points "
            f"expected. Consider re-shooting at higher fps."
        )
        return False, msg
    return True, None


def _check_stability(frame_paths: list[Path], sample: int = 30) -> tuple[bool, float]:
    """Estimate camera motion by comparing background shift between sampled frames."""
    step = max(1, len(frame_paths) // sample)
    sampled = frame_paths[::step][:sample]

    prev_gray = None
    displacements: list[float] = []

    for p in sampled:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
            # Mean absolute difference as a proxy for global displacement
            displacements.append(float(diff.mean()))
        prev_gray = gray

    if not displacements:
        return True, 0.0

    score = float(np.mean(displacements))
    ok = score < _STABILITY_THRESHOLD_PX
    return ok, round(score, 2)


def _check_player_visibility(
    frame_paths: list[Path], sample: int = 20
) -> tuple[str, str | None]:
    """Use a lightweight HOG person detector to estimate player coverage."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    step = max(1, len(frame_paths) // sample)
    sampled = frame_paths[::step][:sample]

    detected = 0
    for p in sampled:
        img = cv2.imread(str(p))
        if img is None:
            continue
        rects, _ = hog.detectMultiScale(
            img, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        if len(rects) > 0:
            detected += 1

    ratio = detected / len(sampled) if sampled else 0.0

    if ratio >= _PLAYER_VISIBILITY_MIN_RATIO:
        return "ok", None
    elif ratio > 0:
        note = (
            f"Player detected in only {ratio:.0%} of sampled frames. "
            "Ensure the player fills enough of the frame and is not occluded."
        )
        return "warning", note
    else:
        note = (
            "No player detected in sampled frames. "
            "Check that the player is visible, large enough, and well-lit. "
            "Fixed tripod position with player occupying at least 1/4 of frame height recommended."
        )
        return "fail", note
