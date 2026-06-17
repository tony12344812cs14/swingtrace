"""Persist VideoMeta to meta.json so downstream modules don't re-read the video."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.models import (
    DominantHand,
    QualityChecks,
    StrokeType,
    VideoMeta,
)


def build_meta(
    raw: dict,
    stroke_type: StrokeType,
    dominant_hand: DominantHand,
    two_handed_backhand: bool,
    checks: QualityChecks,
    frame_timestamps: list[float],
    frames_dir: Path,
    source_path: Path,
) -> VideoMeta:
    source_hash = _sha256(source_path)
    return VideoMeta(
        fps_real=raw["fps_real"],
        is_vfr=raw["is_vfr"],
        frame_timestamps=frame_timestamps,
        width=raw["width"],
        height=raw["height"],
        total_frames=raw["total_frames"],
        duration=raw["duration"],
        stroke_type=stroke_type,
        dominant_hand=dominant_hand,
        two_handed_backhand=two_handed_backhand,
        checks=checks,
        source_hash=source_hash,
        frames_dir=frames_dir,
    )


def write_meta(meta: VideoMeta, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialise(meta)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_meta(meta_path: Path) -> VideoMeta:
    data = json.loads(meta_path.read_text())
    checks = QualityChecks(**data["checks"])
    data["checks"] = checks
    data["stroke_type"] = StrokeType(data["stroke_type"])
    data["dominant_hand"] = DominantHand(data["dominant_hand"])
    data["frames_dir"] = Path(data["frames_dir"])
    return VideoMeta(**data)


def _serialise(meta: VideoMeta) -> dict:
    d = asdict(meta)
    d["stroke_type"] = meta.stroke_type.value
    d["dominant_hand"] = meta.dominant_hand.value
    d["frames_dir"] = str(meta.frames_dir)
    return d


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return f"sha256:{h.hexdigest()}"
