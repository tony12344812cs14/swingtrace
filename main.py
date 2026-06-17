#!/usr/bin/env python3
"""Tennis swing analysis — CLI entry point.

Usage:
    python main.py video.mp4 --output data/run1
    python main.py video.mp4 --stroke serve --hand right --output data/run1
    python main.py video.mp4 --racket-head --scale 320  # 320 px = 1 metre
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.models import DominantHand, StrokeType
from src import pipeline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse tennis swing from an MP4 video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", type=Path, help="Path to input .mp4")
    p.add_argument(
        "--output", "-o", type=Path, default=Path("data/output"),
        help="Output directory (default: data/output)",
    )
    p.add_argument(
        "--stroke",
        choices=[s.value for s in StrokeType],
        default=StrokeType.UNKNOWN.value,
        help="Stroke type (default: unknown)",
    )
    p.add_argument(
        "--hand",
        choices=["left", "right"],
        default="right",
        help="Dominant hand (default: right)",
    )
    p.add_argument(
        "--two-handed-backhand", action="store_true",
        help="Player uses a two-handed backhand",
    )
    p.add_argument(
        "--racket-head", action="store_true",
        help="Use extrapolated racket-head position instead of wrist",
    )
    p.add_argument(
        "--scale", type=float, default=None, metavar="PX_PER_METRE",
        help=(
            "Pixels per metre for a known object in frame (e.g. racket ≈ 0.69m). "
            "Enables m/s and km/h speed output. Omit if unavailable."
        ),
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run all stages even if cached outputs exist",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.video.exists():
        logging.error("Video file not found: %s", args.video)
        return 1

    trajectory = pipeline.run(
        video_path=args.video,
        output_dir=args.output,
        stroke_type=StrokeType(args.stroke),
        dominant_hand=DominantHand(args.hand),
        two_handed_backhand=args.two_handed_backhand,
        prefer_racket_head=args.racket_head,
        scale_px_per_metre=args.scale,
        force_rerun=args.force,
    )

    n_observed = sum(1 for p in trajectory.points if p.is_observed)
    n_repaired = sum(1 for p in trajectory.points if p.is_repaired)
    n_total = len(trajectory.points)

    print(
        f"\nDone. {n_total} frames processed — "
        f"{n_observed} observed, {n_repaired} repaired.\n"
        f"Output: {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
