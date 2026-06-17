"""Persist and reload per-frame joint data as JSON.

Stored to disk so module 3 can be re-run without re-running MediaPipe.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.models import FrameJoints, JointCoord, RacketHeadApprox


def write_pose_data(frames: list[FrameJoints], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_serialise_frame(f) for f in frames]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_pose_data(pose_path: Path) -> list[FrameJoints]:
    raw = json.loads(pose_path.read_text())
    return [_deserialise_frame(d) for d in raw]


def _serialise_frame(fj: FrameJoints) -> dict:
    d: dict = {
        "frame_idx": fj.frame_idx,
        "timestamp_sec": fj.timestamp_sec,
        "valid": fj.valid,
        "joints": {k: asdict(v) for k, v in fj.joints.items()},
    }
    if fj.wrist_primary is not None:
        d["wrist_primary"] = asdict(fj.wrist_primary)
    if fj.racket_head_approx is not None:
        d["racket_head_approx"] = asdict(fj.racket_head_approx)
    return d


def _deserialise_frame(d: dict) -> FrameJoints:
    joints = {k: JointCoord(**v) for k, v in d.get("joints", {}).items()}
    wrist = JointCoord(**d["wrist_primary"]) if d.get("wrist_primary") else None
    racket = RacketHeadApprox(**d["racket_head_approx"]) if d.get("racket_head_approx") else None
    return FrameJoints(
        frame_idx=d["frame_idx"],
        timestamp_sec=d["timestamp_sec"],
        valid=d["valid"],
        joints=joints,
        wrist_primary=wrist,
        racket_head_approx=racket,
    )
