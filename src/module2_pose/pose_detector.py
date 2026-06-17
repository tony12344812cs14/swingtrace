"""MediaPipe Pose wrapper.

Responsibilities:
- Run pre-trained MediaPipe Pose on each frame (no fine-tuning needed for body joints).
- Return shoulder-elbow-wrist chain coordinates + confidence per frame.
- Multi-person handling: lock on the largest detected person.
- Low-confidence frames are KEPT (not dropped) for module 3 to repair.

NOTE: This module only answers "where are the joints" — stroke segmentation,
smoothing, and speed calculation belong to module 3.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

from src.models import (
    DominantHand,
    FrameJoints,
    JointCoord,
    StrokeType,
    VideoMeta,
)
from src.module2_pose.joint_selector import select_primary_wrist, extrapolate_racket_head

# MediaPipe landmark indices
_MP_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Minimum visibility threshold to consider a landmark detected
_MIN_VISIBILITY = 0.5


class PoseDetector:
    """Wraps MediaPipe Pose for tennis swing joint extraction."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frames(self, meta: VideoMeta) -> list[FrameJoints]:
        """Process all extracted frames and return per-frame joint data."""
        frame_paths = sorted(meta.frames_dir.glob("*.png"))
        results: list[FrameJoints] = []

        for idx, frame_path in enumerate(frame_paths):
            ts = meta.frame_timestamps[idx] if idx < len(meta.frame_timestamps) else idx / meta.fps_real
            fj = self._process_single_frame(frame_path, idx, ts, meta)
            results.append(fj)

        return results

    def _process_single_frame(
        self,
        frame_path: Path,
        frame_idx: int,
        timestamp_sec: float,
        meta: VideoMeta,
    ) -> FrameJoints:
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            return FrameJoints(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                valid=False,
                joints={},
            )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(img_rgb)

        if not result.pose_landmarks:
            return FrameJoints(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                valid=False,
                joints={},
            )

        h, w = img_bgr.shape[:2]
        joints = _extract_joints(result.pose_landmarks.landmark, w, h)

        # Mark as invalid if all primary wrist joints are low confidence
        primary_wrist = select_primary_wrist(joints, meta.dominant_hand, meta.stroke_type, meta.two_handed_backhand)
        valid = primary_wrist is not None and primary_wrist.confidence >= _MIN_VISIBILITY

        racket_head = None
        if primary_wrist is not None:
            elbow_key = f"{meta.dominant_hand.value}_elbow"
            elbow = joints.get(elbow_key)
            if elbow is not None:
                racket_head = extrapolate_racket_head(primary_wrist, elbow)

        return FrameJoints(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            valid=valid,
            joints=joints,
            wrist_primary=primary_wrist,
            racket_head_approx=racket_head,
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def _extract_joints(
    landmarks, width: int, height: int
) -> dict[str, JointCoord]:
    joints: dict[str, JointCoord] = {}
    for name, idx in _MP_LANDMARKS.items():
        lm = landmarks[idx]
        joints[name] = JointCoord(
            x=lm.x * width,
            y=lm.y * height,
            z=lm.z,
            confidence=lm.visibility,
        )
    return joints
