"""Tennis swing analysis pipeline orchestrator.

Runs all three modules in sequence and caches intermediate results to disk.
Re-running with the same output directory will skip already-completed stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.models import DominantHand, StrokeType, Trajectory
from src.module1_input.video_processor import read_video_meta_raw
from src.module1_input.frame_extractor import extract_frames
from src.module1_input.quality_checker import run_quality_checks
from src.module1_input.meta_writer import build_meta, load_meta, write_meta
from src.module2_pose.pose_detector import PoseDetector
from src.module2_pose.pose_writer import load_pose_data, write_pose_data
from src.module3_trajectory.trajectory_linker import link_trajectory
from src.module3_trajectory.gap_filler import fill_gaps
from src.module3_trajectory.smoother import smooth_trajectory
from src.module3_trajectory.speed_estimator import estimate_speed
from src.module3_trajectory.visualizer import overlay_trajectory, write_trajectory_json

log = logging.getLogger(__name__)


def run(
    video_path: Path,
    output_dir: Path,
    stroke_type: StrokeType = StrokeType.UNKNOWN,
    dominant_hand: DominantHand = DominantHand.RIGHT,
    two_handed_backhand: bool = False,
    prefer_racket_head: bool = False,
    scale_px_per_metre: Optional[float] = None,
    force_rerun: bool = False,
) -> Trajectory:
    """Run the full pipeline. Returns the final Trajectory.

    Intermediate files (frames, meta.json, pose.json) are persisted so that
    only changed downstream stages need to be re-run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    meta_path = output_dir / "meta.json"
    pose_path = output_dir / "pose.json"

    # ── Module 1: Input processing ──────────────────────────────────────────
    if force_rerun or not meta_path.exists():
        log.info("[M1] Reading video metadata…")
        raw = read_video_meta_raw(video_path)

        log.info("[M1] Extracting frames…")
        timestamps = extract_frames(frames_dir, frames_dir, raw["is_vfr"], raw["fps_real"])

        log.info("[M1] Running quality checks…")
        checks = run_quality_checks(
            frames_dir=frames_dir,
            fps_real=raw["fps_real"],
            stroke_type=stroke_type.value,
            total_frames=raw["total_frames"],
        )

        _log_checks(checks)

        meta = build_meta(
            raw=raw,
            stroke_type=stroke_type,
            dominant_hand=dominant_hand,
            two_handed_backhand=two_handed_backhand,
            checks=checks,
            frame_timestamps=timestamps,
            frames_dir=frames_dir,
            source_path=video_path,
        )
        write_meta(meta, meta_path)
        log.info("[M1] meta.json written → %s", meta_path)
    else:
        log.info("[M1] Skipping — meta.json already exists.")
        meta = load_meta(meta_path)
        meta.frames_dir = frames_dir

    # ── Module 2: Pose detection ────────────────────────────────────────────
    if force_rerun or not pose_path.exists():
        log.info("[M2] Running MediaPipe Pose on %d frames…", meta.total_frames)
        with PoseDetector() as detector:
            frame_joints = detector.process_frames(meta)
        write_pose_data(frame_joints, pose_path)
        log.info("[M2] pose.json written → %s", pose_path)
    else:
        log.info("[M2] Skipping — pose.json already exists.")
        frame_joints = load_pose_data(pose_path)

    # ── Module 3: Trajectory output ─────────────────────────────────────────
    log.info("[M3] Linking trajectory…")
    raw_points = link_trajectory(frame_joints, meta, prefer_racket_head=prefer_racket_head)

    log.info("[M3] Filling gaps (bidirectional)…")
    repaired_points = fill_gaps(raw_points)

    log.info("[M3] Smoothing (preserving inflections)…")
    smooth_points = smooth_trajectory(repaired_points)

    log.info("[M3] Estimating speed…")
    final_points = estimate_speed(smooth_points, meta, scale_px_per_metre=scale_px_per_metre)

    trajectory = Trajectory(meta=meta, points=final_points)

    log.info("[M3] Rendering overlay video…")
    video_out = overlay_trajectory(trajectory, output_dir)
    log.info("[M3] Video → %s", video_out)

    json_out = write_trajectory_json(trajectory, output_dir)
    log.info("[M3] JSON → %s", json_out)

    return trajectory


def _log_checks(checks) -> None:
    if not checks.fps_ok:
        log.warning("FPS WARNING: %s", checks.fps_warning)
    if not checks.stability_ok:
        log.warning(
            "STABILITY WARNING: mean frame diff=%.1fpx (threshold 5px). "
            "Hand-held footage will degrade joint tracking.",
            checks.stability_score,
        )
    if checks.player_visibility != "ok":
        log.warning("PLAYER VISIBILITY %s: %s", checks.player_visibility.upper(), checks.player_visibility_note)
